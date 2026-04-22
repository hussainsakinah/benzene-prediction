# Benzene Content Prediction — Air Quality UCI Dataset

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hussainsakinah/benzene-prediction/main?filepath=notebooks/benzene_prediction.ipynb)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/hussainsakinah/benzene-prediction/blob/main/notebooks/benzene_prediction.ipynb)

Predict the concentration of **benzene (C6H6)** in urban air using machine learning regression models trained on the [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

---

## Project Structure

```
benzene-prediction/
├── notebooks/
│   └── benzene_prediction.ipynb   # Full exploratory notebook
├── src/
│   └── benzene_prediction.py      # Clean, runnable Python script
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip)
- **Records:** ~9,300 hourly readings (after cleaning)
- **Target variable:** `C6H6(GT)` — True hourly averaged benzene concentration (µg/m³)
- **Features:** CO, NOx, NO2, O3 sensor readings, temperature, humidity, and more

The dataset is downloaded automatically at runtime — no manual download needed.

---

## Preprocessing

| Step | Detail |
|------|--------|
| Missing values | Encoded as `-200` → replaced with `NaN` → rows dropped |
| Datetime | Parsed from `Date` + `Time` columns, then dropped |
| Skewness correction | Box-Cox transform applied to columns with \|skew\| > 1 (`CO(GT)`, `NMHC(GT)`) |
| Scaling | `StandardScaler` applied to features |
| Split | 70% train / 30% test |

---

## Models

| Model | Tuning Method |
|-------|--------------|
| Linear Regression | Baseline |
| Lasso | GridSearchCV (degree, alpha) |
| LassoCV | Built-in CV over alpha range |
| Ridge | GridSearchCV (degree, alpha) |
| RidgeCV | Built-in CV over alpha range |
| ElasticNet | GridSearchCV (degree, alpha, l1_ratio) |
| ElasticNetCV | Built-in CV over alpha range |

All GridSearch models use a `Pipeline` of `PolynomialFeatures → StandardScaler → Model` with 5-fold cross-validation.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/benzene-prediction.git
cd benzene-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned because some Anaconda builds of pandas are compiled against NumPy 1.x.

### 3a. Run the Python script

```bash
python src/benzene_prediction.py
```

### 3b. Open the notebook

```bash
jupyter notebook notebooks/benzene_prediction.ipynb
```

---

## Evaluation Metrics

Each model is evaluated on the held-out test set using:
- **MSE** — Mean Squared Error
- **R²** — Coefficient of Determination

---

## Bug Fixes (vs. original submission)

Three copy-paste bugs were corrected in the evaluation cells:

| Location | Bug | Fix |
|----------|-----|-----|
| RidgeCV metrics | Used `y_pr_bc_lcv` (LassoCV predictions) | Changed to `y_pr_bc_rcv` |
| ElasticNet R² | Used `y_pr_bc` (Linear Regression predictions) | Changed to `y_pr_bc_en` |
| ElasticNetCV MSE | Used `y_pr_bc_lcv` (LassoCV predictions) | Changed to `y_pr_bc_ecv` |

---

## Author

**Sakinah Hussain** — Assignment 2
🔗 [github.com/hussainsakinah](https://github.com/hussainsakinah)
#
