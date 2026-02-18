# Fruit360 Classification: ML Pipeline & Robustness Analysis

A comprehensive machine learning project for fruit image classification with focus on **robustness testing** under various degradation scenarios (noise, blur, occlusion, lighting changes). This repository explores classical ML approaches (PCA + SVM, KNN, Random Forest) using the **Fruit360-100x100** dataset.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [V1 vs V2: Key Differences](#v1-vs-v2-key-differences)
5. [How to Execute](#how-to-execute)
6. [Pipeline Overview](#pipeline-overview)
7. [Additional Notebooks](#additional-notebooks)
8. [Results & Artifacts](#results--artifacts)

---

## Overview

### Main Objectives

- **Fruit Classification**: Multi-class classification of fruit from the Fruit360-100x100 dataset
- **Robustness Analysis**: Evaluate model performance under realistic degradation scenarios
- **Feature Engineering**: Compare classical features (color histograms, HOG, LBP, GLCM, Gabor)
- **Hyperparameter Tuning**: Systematic optimization via cross-validation
- **Realistic Testing**: Mixed degradation distributions to simulate real-world conditions

### Degradation Scenarios

- **Scenario A (Blur + Noise + Dirt)**: Simulates dusty/blurry conditions with Gaussian blur, noise, and dirt patches
- **Scenario B (Lighting Variation)**: Simulates extreme lighting (dark/overexposed) with noise
- **Scenario C (Occlusion + Bruising)**: Simulates physical damage with occlusion and bruise patches

### Dataset

- **Source**: [Fruit360-100x100](https://github.com/fruits-360/fruits-360-100x100)
- **Size**: ~173k images (~130k training, ~43k test)
- **Classes**: ~79 fruit classes and ~250 fruit varieties
- **Format**: 100×100 RGB images

---

## Installation & Setup

### Step 1: Clone Repository

```bash
cd ~/Documents/repo
git clone <repository-url>
cd Ml_Project
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or conda
conda create -n fruit360 python=3.10
conda activate fruit360
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies** (see `requirements.txt`):
- `numpy`, `pandas` — Data manipulation
- `torch`, `torchvision` — Deep learning & transforms
- `scikit-learn` — Classical ML (SVM, KNN, PCA, etc.)
- `opencv-python` — Image processing
- `scikit-image` — Advanced feature extraction (HOG, LBP, GLCM, Gabor)
- `matplotlib`, `seaborn` — Visualization
- `Pillow` — Image I/O

---

## ⚡ Quick Start

```bash
jupyter notebook FINAL_NOTEBOOK.ipynb
```

This notebook:
1. Downloads dataset (if needed)
2. Creates train/val/test splits (70/30 rule, then 70% train)
3. Applies data augmentation (20% of training set with Scenarios A/B/C)
4. Extracts color histogram features (32 bins per channel)
5. Performs PCA variance analysis & 3-fold CV tuning
6. Trains final PCA + SVM model
7. Evaluates on clean and mixed degraded test sets
8. Saves models to `V2/saved_models/`
9. Generates confusion matrices & classification reports

**Runtime:** ~45-60 minutes

---

## V1 vs V2: Key Differences

| Aspect | V1 (Old) | V2 (Current) |
|--------|----------|--------------|
| **Dataset** | Hand-made (scraping from internet), ~3600 images, poor quality | Fruit360-100x100 (~173k images, public dataset) |
| **Classes** | Limited fruit types, inconsistent labeling | ~79 classes varieties, consistent taxonomy |
| **Performance** | ~75% low accuracy, unreliable  | ~96% clean accuracy, robust to degradations |
| **Structure** | Ad-hoc experimental notebooks (`provaML*.ipynb`) | Organized pipeline with shared utilities (`utils/pipeline_utils.py`) |

**Why we switched:** V1's hand-made dataset was too small and inconsistent for meaningful results. V2 adopts the industry-standard Fruit360 dataset with systematic augmentation, enabling robust model training and comprehensive degradation analysis.

---

## How to Execute

### Setup Environment

```bash
cd ~/Documents/repo/Ml_Project
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Run Main Pipeline

```bash
jupyter notebook FINAL_NOTEBOOK.ipynb
```

Then **Run All Cells** in the notebook.

**Expected Runtime:** 45-60 minutes  
**Output:** Trained models saved to `V2/saved_models/`, confusion matrices, and accuracy reports

---

## Pipeline Overview

```
Raw Images (100×100 RGB)
    ↓
FruitFolderDataset (PyTorch Dataset)
    ↓
Augmentation (Scenarios A/B/C on 20% of training)
    ↓
Feature Extraction (Color histograms: 32 bins × 3 channels = 96 features)
    ↓
StandardScaler normalization
    ↓
PCA dimensionality reduction
    ↓
SVM classification (RBF kernel)
    ↓
Evaluation (accuracy, confusion matrix, per-class metrics)
```

---

## Additional Notebooks

### Exploratory Data Analysis

**[eda_fruit360.ipynb](eda_fruit360.ipynb)** — Comprehensive EDA of the Fruit360 dataset:

- Dataset integrity check (corrupted files, class consistency)
- Class distribution and imbalance analysis
- Visual sampling with image grids
- RGB channel statistics and color dominance
- Duplicate detection using perceptual hashing
- Image quality metrics (blur via Laplacian variance, noise estimation)
- Outlier detection (size anomalies, non-RGB images)
- Stratified split creation for exploratory analysis

**Runtime:** ~10-15 minutes  
**When to use:** First-time dataset exploration or quality assessment

### V2 Specialized Experiments

**V2 Directory** contains focused experiments:

- **Exploration**: `report_eda.ipynb`, `noise_analysis.ipynb`, `unsupervised_degradation_analysis.ipynb`
- **Model Variants**: `pca_svm_*.ipynb`, `knn_svm_*.ipynb`, `lda_svm_*.ipynb`, `stacking_*.ipynb`
- **Robustness Testing**: `pca_svm_realistic_test_set.ipynb`, `combo_noise_test_full.ipynb`
- **Comparison**: `UNIFIED_COMPARISON.ipynb` (side-by-side model evaluation)

**V1 Directory** contains early experimental work — reference only.

---

## Results & Artifacts

**Saved Models** in `V2/saved_models/`:
- `scaler_*.joblib` — StandardScaler for normalization
- `pca_*.joblib` — PCA transformation
- `svm_*.joblib` — Final SVM classifier

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Dataset not found** | Re-run download cell in notebook (auto-clones from GitHub) |
| **Out of memory** | Reduce `BATCH_SIZE` (128 → 64 or 32) |
| **Slow execution** | Lower `AUG_RATIO` (0.20 → 0.10) |

---

## Key Hyperparameters

```python
SIZE = 32                           # Image resize dimension
VARIETY = False                     # False: macro (Apple), True: fine-grained (Apple Braeburn)
BATCH_SIZE = 128                    # DataLoader batch
AUG_RATIO = 0.20                    # 20% of training set augmented
AUG_DIST = {'A': 0.4, 'B': 0.4, 'C': 0.2}  # Scenario distribution
CV_FOLDS = 3                        # Cross-validation folds
HIST_BINS = 32                      # Histogram bins per channel
C_VALUES = [10, 50, 100]            # SVM C search space
GAMMA_VALUES = [0.01, 0.001]        # SVM gamma search space
VARIANCE_TARGETS = [0.80, 0.90, 0.95]  # PCA variance thresholds
```

---

## References

- **Dataset:** [Fruit360-100x100](https://github.com/fruits-360/fruits-360-100x100)
- **PyTorch:** [pytorch.org](https://pytorch.org)
- **scikit-learn:** [scikit-learn.org](https://scikit-learn.org)

---

## Notes

- **V1 is deprecated** — All current work is in V2
- **FINAL_NOTEBOOK** — Complete pipeline in one file
- **Reproducibility** — Set `RANDOM_STATE = 42` for consistent results