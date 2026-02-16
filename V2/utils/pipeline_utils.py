import json
import hashlib
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from torch.utils.data import Dataset


def download_dataset(root_dir, train_dir, test_dir, github_repo, clone_dir):
    os.makedirs(os.path.dirname(root_dir), exist_ok=True)
    subprocess.run(["git", "clone", github_repo, clone_dir], check=True)
    os.makedirs(root_dir, exist_ok=True)
    shutil.move(os.path.join(clone_dir, "Training"), train_dir)
    shutil.move(os.path.join(clone_dir, "Test"), test_dir)
    shutil.rmtree(clone_dir, ignore_errors=True)


class FruitFolderDataset(Dataset):
    """
    Dataset to load and preprocess the fruit images from folder structure.

    root_dir: path to Training/ or Test/
    variety:  False -> macro label (Apple, Banana, ...)
              True  -> fine-grained label (Apple Braeburn, ...)
    """

    def __init__(self, root_dir, transform=None, variety=False):
        self.root_dir = root_dir
        self.transform = transform
        self.variety = variety
        self.samples = []

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            label_str = class_name if variety else class_name.split()[0]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(class_dir, img_name), label_str))

        self.labels = sorted({lbl for _, lbl in self.samples})
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

        print(f"{os.path.basename(root_dir)} -> {len(self.samples)} images, {len(self.labels)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image

        label_idx = self.label_to_idx[label_str]
        return img, label_idx


class AugmentedFruitDataset(Dataset):
    """
    Wrapper che applica augmentation random durante il training.

    Args:
        base_dataset: Dataset originale (es. train_dataset_fg)
        augment_prob: Probabilità di applicare augmentation (0.0-1.0)
        max_augmentations: Numero massimo di degradazioni da applicare
        use_scenarios: Se True, usa scenari A/B/C; se False, usa degradazioni singole
        scenario_weights: Lista di pesi per scenari [A, B, C] (opzionale)
    """

    def __init__(
        self,
        base_dataset,
        augment_prob=0.5,
        max_augmentations=1,
        use_scenarios=False,
        scenario_weights=None,
    ):
        self.base_dataset = base_dataset
        self.augment_prob = augment_prob
        self.max_augmentations = max_augmentations
        self.use_scenarios = use_scenarios

        self.single_augmentations = [
            blur_medium,
            noise_mild,
            dark_mild,
            overexposed_mild,
            dirty_mild,
            bruised_mild,
            occlusion_small,
        ]

        self.scenarios = [scenarioA, scenarioB, scenarioC]
        self.scenario_weights = None
        if scenario_weights is not None:
            weights = np.array(scenario_weights, dtype=np.float32)
            if len(weights) != len(self.scenarios):
                raise ValueError("scenario_weights deve avere lunghezza 3 (A,B,C).")
            if weights.sum() <= 0:
                raise ValueError("scenario_weights deve avere somma > 0.")
            self.scenario_weights = (weights / weights.sum()).tolist()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        if np.random.rand() < self.augment_prob:
            if self.use_scenarios:
                scenario_fn = np.random.choice(self.scenarios, p=self.scenario_weights)
                img = scenario_fn(img)
            else:
                n_augs = np.random.randint(1, self.max_augmentations + 1)
                for _ in range(n_augs):
                    aug_fn = random.choice(self.single_augmentations)
                    img = aug_fn(img)

        return img, label


class AugmentedDatasetWrapper(AugmentedFruitDataset):
    def __init__(self, base_dataset, **kwargs):
        super().__init__(base_dataset, **kwargs)
        source = base_dataset
        if hasattr(base_dataset, "dataset"):
            source = base_dataset.dataset
        self.label_to_idx = getattr(source, "label_to_idx", None)
        self.idx_to_label = getattr(source, "idx_to_label", None)
        self.labels = getattr(source, "labels", None)


def dataloader_to_numpy(loader):
    x_list = []
    y_list = []
    for batch_x, batch_y in loader:
        if isinstance(batch_x, torch.Tensor):
            x_list.append(batch_x.detach().cpu())
        else:
            x_list.append(torch.tensor(batch_x))
        if isinstance(batch_y, torch.Tensor):
            y_list.append(batch_y.detach().cpu())
        else:
            y_list.append(torch.tensor(batch_y))
    if not x_list:
        raise ValueError("Empty loader: no samples found.")
    X = torch.cat(x_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    return X, y


def _short_hash(payload):
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:8]


def make_run_id(meta):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{_short_hash(meta)}"


def save_checkpoint(model, scaler, meta, run_dir=None, save_meta=False):
    meta = dict(meta)
    ckpt_root = Path("artifacts/checkpoints")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id(meta)
    run_dir = Path(run_dir) if run_dir else (
        ckpt_root / meta["task"] / meta["feature"] / meta["model"] / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.joblib")
    if scaler is not None:
        joblib.dump(scaler, run_dir / "scaler.joblib")
    if save_meta:
        meta["saved_at"] = datetime.now().isoformat()
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return run_dir


def color_hist_features(X, bins=16, img_shape=(3, 64, 64)):
    n_samples = X.shape[0]
    feats = np.zeros((n_samples, 3 * bins), dtype=np.float32)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(n_samples):
        img = X[i].reshape(img_shape)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        img_hsv = (img * 255.0).astype(np.uint8)
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)
        hists = []
        for channel in (h, s, v):
            ch_norm = channel.astype(np.float32) / 255.0
            hist, _ = np.histogram(ch_norm.ravel(), bins=bin_edges, density=True)
            hists.append(hist)
        feats[i] = np.concatenate(hists)
    return feats


def gray_hist_features(X, bins=16, img_shape=(1, 64, 64)):
    n_samples = X.shape[0]
    feats = np.zeros((n_samples, bins), dtype=np.float32)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(n_samples):
        img = X[i].reshape(img_shape)
        img = np.clip(img, 0.0, 1.0)
        ch_norm = img.astype(np.float32).ravel()
        hist, _ = np.histogram(ch_norm, bins=bin_edges, density=True)
        feats[i] = hist
    return feats


def clamp01(x):
    return torch.clamp(x, 0.0, 1.0)


def add_color_patches(
    x,
    num_patches,
    color,
    alpha_range=(0.4, 0.7),
    size_range=(0.05, 0.15),
):
    _, H, W = x.shape
    out = x.clone()
    for _ in range(num_patches):
        s = np.random.uniform(size_range[0], size_range[1])
        patch_area = s * H * W / 4
        r = np.random.uniform(0.5, 1.5)
        patch_h = int(np.sqrt(patch_area * r))
        patch_w = int(np.sqrt(patch_area / r))
        patch_h = max(1, min(H, patch_h))
        patch_w = max(1, min(W, patch_w))
        top = np.random.randint(0, H - patch_h + 1)
        left = np.random.randint(0, W - patch_w + 1)
        bottom = top + patch_h
        right = left + patch_w
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
        patch = out[:, top:bottom, left:right]
        blended = alpha * color + (1 - alpha) * patch
        out[:, top:bottom, left:right] = blended
    return clamp01(out)


def add_occlusion_patch(
    x,
    area_ratio=0.1,
    color=torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1),
    alpha=0.5,
):
    _, H, W = x.shape
    out = x.clone()
    patch_area = area_ratio * H * W
    r = np.random.uniform(0.5, 1.5)
    patch_h = int(np.sqrt(patch_area * r))
    patch_w = int(np.sqrt(patch_area / r))
    patch_h = max(1, min(H, patch_h))
    patch_w = max(1, min(W, patch_w))
    top = np.random.randint(0, H - patch_h + 1)
    left = np.random.randint(0, W - patch_w + 1)
    bottom = top + patch_h
    right = left + patch_w
    patch = out[:, top:bottom, left:right]
    blended = alpha * color + (1 - alpha) * patch
    out[:, top:bottom, left:right] = blended
    return clamp01(out)


color_dirt = torch.tensor([0.3, 0.25, 0.2]).view(3, 1, 1)
color_bruise = torch.tensor([0.25, 0.2, 0.15]).view(3, 1, 1)


def noise_mild(x):
    return clamp01(x + torch.randn_like(x) * 0.025)


def dark_mild(x):
    return clamp01(x * 0.65)


def overexposed_mild(x):
    return clamp01(x * 1.35)


def dirty_mild(x):
    return add_color_patches(
        x,
        num_patches=2,
        color=color_dirt,
        alpha_range=(0.5, 0.8),
        size_range=(0.03, 0.08),
    )


def bruised_mild(x):
    return add_color_patches(
        x,
        num_patches=1,
        color=color_bruise,
        alpha_range=(0.4, 0.7),
        size_range=(0.03, 0.08),
    )


def occlusion_small(x):
    return add_occlusion_patch(x, area_ratio=0.10, alpha=0.5)


blur_medium = T.GaussianBlur(kernel_size=5, sigma=1.0)


def scenarioA(x):
    x = blur_medium(x)
    x = noise_mild(x)
    if np.random.rand() < 0.7:
        x = dirty_mild(x)
    return x


def scenarioB(x):
    if np.random.rand() < 0.5:
        x = dark_mild(x)
    else:
        x = overexposed_mild(x)
    x = noise_mild(x)
    return x


def scenarioC(x):
    x = occlusion_small(x)
    if np.random.rand() < 0.5:
        x = bruised_mild(x)
    else:
        x = dirty_mild(x)
    return x


def _prepare_img(Xi, img_shape):
    img = Xi.reshape(img_shape)
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    return img


def _to_gray(img):
    return rgb2gray(img)


def hog_features(
    img_gray,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    orientations=9,
):
    return hog(
        img_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )


def lbp_features(img_gray, P=8, R=1):
    lbp = local_binary_pattern(img_gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    return hist


def glcm_features(
    img_gray,
    distances=(1, 2),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
):
    img_u8 = np.clip(img_gray * 255.0, 0, 255).astype(np.uint8)
    glcm = graycomatrix(
        img_u8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = [graycoprops(glcm, p).ravel() for p in props]
    return np.concatenate(feats)


def gabor_features(
    img_gray,
    frequencies=(0.1, 0.2, 0.3),
    thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
):
    feats = []
    for freq in frequencies:
        for theta in thetas:
            real, imag = gabor(img_gray, frequency=freq, theta=theta)
            mag = np.sqrt(real ** 2 + imag ** 2)
            feats.append(mag.mean())
            feats.append(mag.var())
    return np.array(feats, dtype=np.float32)


def compute_feature_blocks(
    X,
    img_shape=(3, 64, 64),
    color_bins=16,
    spatial_grid=(2, 2),
    hog_params=None,
    lbp_params=None,
    glcm_params=None,
    gabor_params=None,
    feature_keys=None,
):
    hog_params = hog_params or {}
    lbp_params = lbp_params or {}
    glcm_params = glcm_params or {}
    gabor_params = gabor_params or {}

    if feature_keys is None:
        feature_keys = {"color_hist", "hog", "lbp", "glcm", "gabor"}
    else:
        feature_keys = set(feature_keys)

    blocks = {}
    if "color_hist" in feature_keys:
        blocks["color_hist"] = color_hist_features(X, bins=color_bins, img_shape=img_shape)

    need_gray = any(k in feature_keys for k in ("hog", "lbp", "glcm", "gabor"))
    if need_gray:
        hog_list, lbp_list, glcm_list, gabor_list = [], [], [], []
        for i in range(X.shape[0]):
            img = _prepare_img(X[i], img_shape)
            gray = _to_gray(img)
            if "hog" in feature_keys:
                hog_list.append(hog_features(gray, **hog_params))
            if "lbp" in feature_keys:
                lbp_list.append(lbp_features(gray, **lbp_params))
            if "glcm" in feature_keys:
                glcm_list.append(glcm_features(gray, **glcm_params))
            if "gabor" in feature_keys:
                gabor_list.append(gabor_features(gray, **gabor_params))
        if "hog" in feature_keys:
            blocks["hog"] = np.vstack(hog_list).astype(np.float32)
        if "lbp" in feature_keys:
            blocks["lbp"] = np.vstack(lbp_list).astype(np.float32)
        if "glcm" in feature_keys:
            blocks["glcm"] = np.vstack(glcm_list).astype(np.float32)
        if "gabor" in feature_keys:
            blocks["gabor"] = np.vstack(gabor_list).astype(np.float32)
    return blocks


def concat_feature_blocks(blocks, keys):
    return np.concatenate([blocks[k] for k in keys], axis=1)
