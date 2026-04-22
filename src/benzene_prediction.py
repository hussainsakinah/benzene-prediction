"""
Benzene Content Prediction in Air - Air Quality UCI Dataset
===========================================================
Predicts C6H6 (benzene) concentration using regression models:
Linear Regression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
"""

import warnings
warnings.filterwarnings('ignore')

import zipfile
import io
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import boxcox

from sklearn.linear_model import (
    LinearRegression, Lasso, LassoCV,
    Ridge, RidgeCV, ElasticNet, ElasticNetCV
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'

print("Downloading dataset...")
response = urllib.request.urlopen(URL)
zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

with zip_file.open('AirQualityUCI.csv') as file:
    data = pd.read_csv(
        file,
        sep=';',
        decimal=',',
        parse_dates={'Datetime': ['Date', 'Time']},
        na_values=-200
    )

print(f"Raw data shape: {data.shape}")


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────

# Drop Datetime and empty trailing columns
data_pd = data.drop(columns=['Datetime', 'Unnamed: 15', 'Unnamed: 16'], errors='ignore')

# Drop rows with any NaN
data_pd = data_pd.dropna()
print(f"After dropping NaN: {data_pd.shape}")

# Skewness check and Box-Cox transform for highly skewed columns (|skew| > 1)
skewness = data_pd.skew()
high_skew_cols = [col for col, v in skewness.items() if not (-1 < v < 1)]
print(f"High skewness columns: {high_skew_cols}")

lambdas = {}
logged = []
for col in high_skew_cols:
    try:
        transformed, lam = boxcox(data_pd[col])
        lambdas[col] = lam
        data_pd[col] = transformed
    except Exception:
        print(f"Column {col} has non-positive values — applying log1p instead.")
        data_pd[col] = np.log1p(data_pd[col])
        logged.append(col)

print(f"Box-Cox lambdas: {lambdas}")
print(f"Log1p-transformed columns: {logged}")


# ─────────────────────────────────────────────
# 3. FEATURES & TARGET
# ─────────────────────────────────────────────

X = data_pd.drop(columns=['C6H6(GT)'])
y = data_pd['C6H6(GT)']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled  = scaler.transform(xtest)

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
kf_bc = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}


# ─────────────────────────────────────────────
# 4. LINEAR REGRESSION
# ─────────────────────────────────────────────

lr = LinearRegression()
lr.fit(xtrain_scaled, ytrain)
y_pred_lr = lr.predict(xtest_scaled)

results['Linear Regression'] = {
    'MSE': mean_squared_error(ytest, y_pred_lr),
    'R2':  r2_score(ytest, y_pred_lr)
}


# ─────────────────────────────────────────────
# 5. LASSO (GridSearchCV)
# ─────────────────────────────────────────────

lasso_pipeline = Pipeline([
    ('poly',   PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('lasso',  Lasso())
])

param_grid_lasso = {
    'poly__degree': [2, 3, 5],
    'lasso__alpha': [0.1, 0.001, 1, 10]
}

grid_lasso = GridSearchCV(lasso_pipeline, param_grid_lasso, cv=kf, scoring='neg_mean_squared_error')
grid_lasso.fit(xtrain, ytrain)

best_lasso = grid_lasso.best_estimator_
y_pred_lasso = best_lasso.predict(xtest)

results['Lasso (GridSearchCV)'] = {
    'MSE':         mean_squared_error(ytest, y_pred_lasso),
    'R2':          r2_score(ytest, y_pred_lasso),
    'best_params': grid_lasso.best_params_
}


# ─────────────────────────────────────────────
# 6. LASSO CV
# ─────────────────────────────────────────────

lcv = LassoCV(alphas=np.geomspace(1e-9, 1e0, 10), cv=kf_bc)
lcv.fit(xtrain, ytrain)
y_pred_lcv = lcv.predict(xtest)

results['LassoCV'] = {
    'MSE': mean_squared_error(ytest, y_pred_lcv),
    'R2':  r2_score(ytest, y_pred_lcv)
}


# ─────────────────────────────────────────────
# 7. RIDGE (GridSearchCV)
# ─────────────────────────────────────────────

ridge_pipeline = Pipeline([
    ('poly',   PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('ridge',  Ridge())
])

param_grid_ridge = {
    'poly__degree': [2, 3, 5],
    'ridge__alpha': [0.1, 0.001, 1, 10]
}

grid_ridge = GridSearchCV(ridge_pipeline, param_grid_ridge, cv=kf, scoring='neg_mean_squared_error')
grid_ridge.fit(xtrain, ytrain)

best_ridge = grid_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(xtest)

results['Ridge (GridSearchCV)'] = {
    'MSE':         mean_squared_error(ytest, y_pred_ridge),
    'R2':          r2_score(ytest, y_pred_ridge),
    'best_params': grid_ridge.best_params_
}


# ─────────────────────────────────────────────
# 8. RIDGE CV
# ─────────────────────────────────────────────

rcv = RidgeCV(alphas=np.geomspace(1e-9, 1e0, 10), cv=kf_bc)
rcv.fit(xtrain, ytrain)
y_pred_rcv = rcv.predict(xtest)

results['RidgeCV'] = {
    'MSE': mean_squared_error(ytest, y_pred_rcv),
    'R2':  r2_score(ytest, y_pred_rcv)
}


# ─────────────────────────────────────────────
# 9. ELASTIC NET (GridSearchCV)
# ─────────────────────────────────────────────

en_pipeline = Pipeline([
    ('pf',     PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('en',     ElasticNet())
])

param_grid_en = {
    'pf__degree':   [2, 3, 5],
    'en__alpha':    [0.1, 0.001, 1, 10],
    'en__l1_ratio': [0.2, 0.5, 0.8]
}

grid_en = GridSearchCV(en_pipeline, param_grid_en, cv=kf, scoring='neg_mean_squared_error')
grid_en.fit(xtrain, ytrain)

best_en = grid_en.best_estimator_
y_pred_en = best_en.predict(xtest)

results['ElasticNet (GridSearchCV)'] = {
    'MSE':         mean_squared_error(ytest, y_pred_en),
    'R2':          r2_score(ytest, y_pred_en),
    'best_params': grid_en.best_params_
}


# ─────────────────────────────────────────────
# 10. ELASTIC NET CV
# ─────────────────────────────────────────────

encv = ElasticNetCV(alphas=np.geomspace(1e-9, 1e0, 10), cv=kf_bc)
encv.fit(xtrain, ytrain)
y_pred_encv = encv.predict(xtest)

results['ElasticNetCV'] = {
    'MSE': mean_squared_error(ytest, y_pred_encv),
    'R2':  r2_score(ytest, y_pred_encv)
}


# ─────────────────────────────────────────────
# 11. RESULTS SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print(f"{'Model':<30} {'MSE':>10} {'R2':>10}")
print("=" * 55)
for model, metrics in results.items():
    print(f"{model:<30} {metrics['MSE']:>10.4f} {metrics['R2']:>10.4f}")
print("=" * 55)
