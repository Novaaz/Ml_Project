# Fruit360 Classification: ML Pipeline & Robustness Analysis

A comprehensive machine learning project for fruit image classification with focus on **robustness testing** under various degradation scenarios (noise, blur, occlusion, lighting changes). This repository explores classical ML approaches (PCA + SVM, KNN, Random Forest) using the **Fruit360-100x100** dataset.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [V1 vs V2: Key Differences](#v1-vs-v2-key-differences)
3. [How to Execute](#how-to-execute)
4. [Additional Notebooks](#additional-notebooks)

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

## Additional Notebooks

### Exploratory Data Analysis

**[eda_fruit360.ipynb](eda_fruit360.ipynb)** — Comprehensive EDA of the Fruit360 dataset

### V2 Specialized Experiments

**V2 Directory** contains focused experiments on which the final notebook is based.
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