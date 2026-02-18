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
- **Type**: Large-scale fruit image dataset with training and test splits
- **Classes**: Multiple fruit types and varieties
- **Format**: RGB images

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

## Quick Start

```bash
jupyter notebook FINAL_NOTEBOOK.ipynb
```

This notebook:
1. Downloads dataset (if needed)
2. Creates train/val/test splits
3. Applies data augmentation with degradation scenarios
4. Extracts color histogram features
5. Performs PCA variance analysis and hyperparameter tuning via cross-validation
6. Trains final PCA + SVM model
7. Evaluates on clean and degraded test sets
8. Saves models and generates performance reports

---

## V1 vs V2: Key Differences

| Aspect | V1 (Old) | V2 (Current) |
|--------|----------|--------------|
| **Dataset** | Hand-made (web scraping), small, poor quality | Fruit360-100x100, large-scale public dataset |
| **Classes** | Limited fruit types, inconsistent labeling | Multiple fruit classes and varieties, consistent taxonomy |
| **Performance** | Low accuracy, unreliable | High accuracy, robust to degradations |
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

**Output:** Trained models saved to `V2/saved_models/`, confusion matrices, and accuracy reports

---

## Pipeline Overview

```
Raw Images (RGB)
    ↓
FruitFolderDataset (PyTorch Dataset)
    ↓
Data Augmentation (Degradation Scenarios A/B/C)
    ↓
Feature Extraction (Color Histograms)
    ↓
StandardScaler Normalization
    ↓
PCA Dimensionality Reduction
    ↓
SVM Classification (RBF Kernel)
    ↓
Evaluation (Accuracy, Confusion Matrix, Per-Class Metrics)
```

---

## Additional Notebooks

### Exploratory Data Analysis

**[eda_fruit360.ipynb](eda_fruit360.ipynb)** — Comprehensive EDA of the Fruit360 dataset

### V2 Specialized Experiments

**V2 Directory** contains focused experiments:

- **Exploration**: `report_eda.ipynb`, `noise_analysis.ipynb`, `unsupervised_degradation_analysis.ipynb`
- **Model Variants**: `pca_svm_*.ipynb`, `knn_svm_*.ipynb`, `lda_svm_*.ipynb`, `stacking_*.ipynb`
- **Robustness Testing**: `pca_svm_realistic_test_set.ipynb`, `combo_noise_test_full.ipynb`
- **Comparison**: `UNIFIED_COMPARISON.ipynb` (side-by-side model evaluation)

**V1 Directory** contains early experimental work — reference only.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory** | Reduce `BATCH_SIZE` parameter |
| **Slow execution** | Lower `AUG_RATIO` parameter |

---

## References

- **Dataset:** [Fruit360-100x100](https://github.com/fruits-360/fruits-360-100x100)
- **PyTorch:** [pytorch.org](https://pytorch.org)
- **scikit-learn:** [scikit-learn.org](https://scikit-learn.org)

---

## Notes

- **V1 is deprecated** — All current work is in V2
- **FINAL_NOTEBOOK** — Complete pipeline in one file
- **Reproducibility** — Use fixed random seed for consistent results
- **Hyperparameters** — Configurable in notebook header (image size, batch size, augmentation ratio, etc.)