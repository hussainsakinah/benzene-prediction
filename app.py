import warnings
warnings.filterwarnings('ignore')

import zipfile
import io
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import (
    LinearRegression, Lasso, LassoCV,
    Ridge, RidgeCV, ElasticNet, ElasticNetCV
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Benzene Predictor",
    page_icon="🌫️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 800; color: #1f77b4; }
    .subtitle   { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .metric-card {
        background: #f0f8ff;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .best-badge {
        background: #d4edda;
        border-left: 5px solid #28a745;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING (raw only — no transforms yet)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw():
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
    response = urllib.request.urlopen(URL)
    zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

    with zip_file.open('AirQualityUCI.csv') as file:
        data = pd.read_csv(
            file, sep=';', decimal=',',
            na_values=-200
        )

    # Combine Date + Time into Datetime manually
    # (parse_dates dict syntax was removed in newer pandas versions)
    if 'Date' in data.columns and 'Time' in data.columns:
        data['Datetime'] = pd.to_datetime(
            data['Date'].astype(str) + ' ' + data['Time'].astype(str),
            dayfirst=True, errors='coerce'
        )

    # Drop metadata and unnamed columns
    data_pd = data.drop(columns=['Date', 'Time', 'Datetime', 'Unnamed: 15', 'Unnamed: 16'], errors='ignore')

    # ── BUG FIX: Remove direct benzene proxies that cause R² ≈ 1.0 ──────────
    # PT08.S2(NMHC) is a metal-oxide sensor tuned specifically to detect
    # non-methane hydrocarbons (i.e. benzene). Correlation with C6H6(GT) ≈ 0.99.
    # NMHC(GT) is the ground-truth NMHC measurement — another direct proxy.
    # Keeping either column makes the prediction task trivial and meaningless.
    leaky_cols = ['PT08.S2(NMHC)', 'NMHC(GT)']
    data_pd = data_pd.drop(columns=leaky_cols, errors='ignore')
    # ─────────────────────────────────────────────────────────────────────────

    data_pd = data_pd.dropna()
    return data_pd


# ─────────────────────────────────────────────
# FIX 1: Skewness computed & BoxCox fitted on
#         TRAIN split only, then applied to test.
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def split_and_preprocess(_data_pd):
    """
    Split first, then fit all transforms (skewness / BoxCox) exclusively on
    the training set, and apply the fitted transforms to the test set.
    This prevents any test-set statistics from leaking into preprocessing.
    """
    X_raw = _data_pd.drop(columns=['C6H6(GT)'])
    y_raw = _data_pd['C6H6(GT)']

    # --- Train/test split BEFORE any fitting ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42
    )

    # --- Fit BoxCox / log transforms on TRAIN only ---
    skewness_train = X_train_raw.skew()   # skewness from train set only
    high_skew_cols = [col for col, v in skewness_train.items() if not (-1 < v < 1)]

    lambdas = {}
    logged = []

    X_train = X_train_raw.copy()
    X_test  = X_test_raw.copy()

    for col in high_skew_cols:
        try:
            transformed_train, lam = boxcox(X_train[col])   # fit on train
            lambdas[col] = lam
            X_train[col] = transformed_train
            # apply same lambda to test (no refit)
            X_test[col]  = boxcox(X_test[col], lmbda=lam)
        except Exception:
            X_train[col] = np.log1p(X_train[col])
            X_test[col]  = np.log1p(X_test[col])
            logged.append(col)

    # Apply same transform to target if needed (check skew on train target)
    y_skew = pd.Series(y_train).skew()
    y_lambda = None
    y_logged = False
    if not (-1 < y_skew < 1):
        try:
            y_train_t, y_lambda = boxcox(y_train)
            y_train = pd.Series(y_train_t, index=y_train.index)
            y_test_t = boxcox(y_test, lmbda=y_lambda)
            y_test  = pd.Series(y_test_t, index=y_test.index)
        except Exception:
            y_train = np.log1p(y_train)
            y_test  = np.log1p(y_test)
            y_logged = True

    return (
        X_train, X_test, y_train, y_test,
        skewness_train, lambdas, logged,
        y_lambda, y_logged,
        X_train_raw.columns.tolist()
    )


# ─────────────────────────────────────────────
# MODEL TRAINING
# FIX 2: LassoCV / RidgeCV / ElasticNetCV now
#         wrapped in Pipelines (scaler inside CV).
# FIX 3: Predict-page LR trained on train set only.
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_all_models(_X_train, _X_test, _y_train, _y_test):
    X_train, X_test   = _X_train, _X_test
    y_train, y_test   = _y_train, _y_test

    # Scaler fitted on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)       # transform only — no refit

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results     = {}
    predictions = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    results['Linear Regression'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²':  r2_score(y_test, y_pred)
    }
    predictions['Linear Regression'] = y_pred

    # 2. Lasso (GridSearchCV) — scaler inside pipeline, CV on train only
    lasso_pipe = Pipeline([
        ('poly',   PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('lasso',  Lasso(max_iter=10000))
    ])
    grid_lasso = GridSearchCV(
        lasso_pipe,
        {'poly__degree': [2, 3, 5], 'lasso__alpha': [0.1, 0.001, 1, 10]},
        cv=kf, scoring='neg_mean_squared_error'
    )
    grid_lasso.fit(X_train, y_train)          # raw train; pipeline scales internally
    y_pred = grid_lasso.best_estimator_.predict(X_test)
    results['Lasso'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²':  r2_score(y_test, y_pred),
        'best_params': grid_lasso.best_params_
    }
    predictions['Lasso'] = y_pred

    # 3. LassoCV — FIX 2: wrapped in Pipeline so scaler is inside CV folds
    lcv_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lcv',    LassoCV(alphas=np.geomspace(1e-9, 1e0, 10), cv=kf, max_iter=10000))
    ])
    lcv_pipe.fit(X_train, y_train)
    y_pred = lcv_pipe.predict(X_test)
    results['LassoCV'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²':  r2_score(y_test, y_pred)
    }
    predictions['LassoCV'] = y_pred

    # 4. Ridge (GridSearchCV)
    ridge_pipe = Pipeline([
        ('poly',   PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('ridge',  Ridge())
    ])
    grid_ridge = GridSearchCV(
        ridge_pipe,
        {'poly__degree': [2, 3, 5], 'ridge__alpha': [0.1, 0.001, 1, 10]},
        cv=kf, scoring='neg_mean_squared_error'
    )
    grid_ridge.fit(X_train, y_train)
    y_pred = grid_ridge.best_estimator_.predict(X_test)
    results['Ridge'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²':  r2_score(y_test, y_pred),
        'best_params': grid_ridge.best_params_
    }
    predictions['Ridge'] = y_pred

    # 5. RidgeCV — FIX 2: wrapped in Pipeline
    rcv_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rcv',    RidgeCV(alphas=np.geomspace(1e-9, 1e0, 10), cv=kf))
    ])
    rcv_pipe.fit(X_train, y_train)
    y_pred = rcv_pipe.predict(X_test)
    results['RidgeCV'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²':  r2_score(y_test, y_pred)
    }
    predictions['RidgeCV'] = y_pred

    # 6. ElasticNet (GridSearchCV)
    en_pipe = Pipeline([
        ('pf',     PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('en',     ElasticNet(max_iter=10000))
    ])
    grid_en = GridSearchCV(
        en_pipe,
        {
            'pf__degree':  [2, 3, 5],
            'en__alpha':   [0.1, 0.001, 1, 10],
            'en__l1_ratio':[0.2, 0.5, 0.8]
        },
        cv=kf, scoring='neg_mean_squared_error'
    )
    grid_en.fit(X_train, y_train)
    y_pred = grid_en.best_estimator_.predict(X_test)
    results['ElasticNet'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²':  r2_score(y_test, y_pred),
        'best_params': grid_en.best_params_
    }
    predictions['ElasticNet'] = y_pred

    # 7. ElasticNetCV — FIX 2: wrapped in Pipeline
    encv_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('encv',   ElasticNetCV(alphas=np.geomspace(1e-9, 1e0, 10), cv=kf, max_iter=10000))
    ])
    encv_pipe.fit(X_train, y_train)
    y_pred = encv_pipe.predict(X_test)
    results['ElasticNetCV'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²':  r2_score(y_test, y_pred)
    }
    predictions['ElasticNetCV'] = y_pred

    # FIX 3: LR for Predict page — trained on TRAIN set only (not full dataset)
    lr_pred = LinearRegression()
    lr_pred.fit(X_train_scaled, y_train)

    return results, predictions, y_test, scaler, X_train.columns.tolist(), X_train, y_train, lr_pred


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## Benzene Predictor")
st.sidebar.markdown("**Dataset:** UCI Air Quality")
st.sidebar.markdown("**Target:** C6H6 (Benzene) µg/m³")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Overview", "Model Results", "Predict Benzene"])
st.sidebar.markdown("---")
st.sidebar.markdown("**By:** Sakinah Hussain")
st.sidebar.markdown("[GitHub Repo](https://github.com/hussainsakinah/benzene-prediction)")

# ─────────────────────────────────────────────
# LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
with st.spinner("Loading data..."):
    data_pd_raw = load_raw()

with st.spinner("Splitting and preprocessing (train-only transforms)..."):
    (
        X_train, X_test, y_train, y_test,
        skewness_train, lambdas, logged,
        y_lambda, y_logged,
        feature_cols
    ) = split_and_preprocess(data_pd_raw)

# Reconstruct a display-friendly version of the processed training data for overview
# y_train may lose its Series name after BoxCox transform — restore it explicitly
y_train_named = y_train.copy()
y_train_named.name = 'C6H6(GT)'
data_pd_display = pd.concat([X_train, y_train_named], axis=1)

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.markdown('<div class="main-title">Benzene Content Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predicting C6H6 concentration in urban air using the UCI Air Quality Dataset</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(data_pd_raw):,}")
    col2.metric("Features", len(data_pd_raw.columns) - 1)
    col3.metric("Target", "C6H6(GT)")
    col4.metric("Models Trained", "7")

    st.markdown('<div class="section-header">Dataset Preview (Training Set, Post-Transform)</div>', unsafe_allow_html=True)
    st.dataframe(data_pd_display.head(10), width='stretch')

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Benzene Distribution (Train)</div>', unsafe_allow_html=True)
        fig = px.histogram(data_pd_display, x='C6H6(GT)', nbins=60, color_discrete_sequence=['#1f77b4'],
                           labels={'C6H6(GT)': 'Benzene (transformed)'})
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown('<div class="section-header">Feature Skewness (Train Only)</div>', unsafe_allow_html=True)
        skew_df = skewness_train.reset_index()
        skew_df.columns = ['Feature', 'Skewness']
        fig2 = px.bar(skew_df, x='Feature', y='Skewness', color='Skewness',
                      color_continuous_scale='RdBu_r')
        fig2.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="|skew|=1")
        fig2.add_hline(y=-1, line_dash="dash", line_color="red")
        fig2.update_layout(margin=dict(t=20))
        st.plotly_chart(fig2, width='stretch')

    st.markdown('<div class="section-header">Correlation Heatmap (Train)</div>', unsafe_allow_html=True)
    corr = data_pd_display.corr()
    fig3 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto')
    fig3.update_layout(margin=dict(t=20))
    st.plotly_chart(fig3, width='stretch')

# ─────────────────────────────────────────────
# PAGE 2 — MODEL RESULTS
# ─────────────────────────────────────────────
elif page == "Model Results":
    st.markdown('<div class="main-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">All 7 regression models trained and evaluated on the Air Quality dataset</div>', unsafe_allow_html=True)

    with st.spinner("Training all models — this takes ~1-2 minutes..."):
        results, predictions, ytest, scaler, feature_cols, xtrain, ytrain, lr_pred = train_all_models(
            X_train, X_test, y_train, y_test
        )

    # Summary table
    summary = []
    for model, m in results.items():
        summary.append({'Model': model, 'MSE': round(m['MSE'], 4), 'R²': round(m['R²'], 4)})
    df_summary = pd.DataFrame(summary).sort_values('R²', ascending=False).reset_index(drop=True)

    best_model = df_summary.iloc[0]['Model']
    st.success(f"Best Model: **{best_model}** — R² = {df_summary.iloc[0]['R²']}")

    st.markdown('<div class="section-header">Results Table</div>', unsafe_allow_html=True)
    st.dataframe(df_summary.style.highlight_max(subset=['R²'], color='#d4edda')
                                  .highlight_min(subset=['MSE'], color='#d4edda'),
                 width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">R² Score by Model</div>', unsafe_allow_html=True)
        fig = px.bar(df_summary, x='Model', y='R²', color='R²',
                     color_continuous_scale='Blues', text='R²')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(showlegend=False, margin=dict(t=20), xaxis_tickangle=-30)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-header">MSE by Model</div>', unsafe_allow_html=True)
        fig2 = px.bar(df_summary.sort_values('MSE'), x='Model', y='MSE', color='MSE',
                      color_continuous_scale='Reds_r', text='MSE')
        fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig2.update_layout(showlegend=False, margin=dict(t=20), xaxis_tickangle=-30)
        st.plotly_chart(fig2, width='stretch')

    # Actual vs Predicted
    st.markdown('<div class="section-header">Actual vs Predicted — Select Model</div>', unsafe_allow_html=True)
    chosen = st.selectbox("Choose a model to inspect:", list(predictions.keys()))
    y_pred_chosen = predictions[chosen]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ytest.values, y=y_pred_chosen, mode='markers',
                              marker=dict(color='#1f77b4', opacity=0.5, size=5),
                              name='Predictions'))
    min_val = min(ytest.min(), y_pred_chosen.min())
    max_val = max(ytest.max(), y_pred_chosen.max())
    fig3.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                              mode='lines', line=dict(color='red', dash='dash'), name='Perfect Fit'))
    fig3.update_layout(xaxis_title='Actual Benzene (transformed)', yaxis_title='Predicted Benzene (transformed)',
                       margin=dict(t=20))
    st.plotly_chart(fig3, width='stretch')

    # Best params
    if 'best_params' in results.get(chosen, {}):
        st.markdown(f"**Best Parameters for {chosen}:**")
        st.json(results[chosen]['best_params'])

# ─────────────────────────────────────────────
# PAGE 3 — PREDICT
# FIX 3: lr_pred is now returned from train_all_models
#         (fitted on train set only, not full data)
# ─────────────────────────────────────────────
elif page == "Predict Benzene":
    st.markdown('<div class="main-title">Predict Benzene Level</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter sensor readings to get a real-time benzene concentration prediction</div>', unsafe_allow_html=True)

    with st.spinner("Training models..."):
        results, predictions, ytest, scaler, feature_cols, xtrain, ytrain, lr_pred = train_all_models(
            X_train, X_test, y_train, y_test
        )

    st.markdown('<div class="section-header">Input Sensor Values</div>', unsafe_allow_html=True)
    st.info("Adjust sliders to match real sensor readings. Default values are training-set averages.")

    # Use TRAIN set stats for slider defaults/bounds (not full dataset)
    defaults = X_train.mean()

    col1, col2, col3 = st.columns(3)
    inputs = {}

    for i, feat in enumerate(feature_cols):
        col = [col1, col2, col3][i % 3]
        mn      = float(X_train[feat].min())
        mx      = float(X_train[feat].max())
        default = float(defaults[feat])
        with col:
            inputs[feat] = st.slider(feat, min_value=mn, max_value=mx, value=default, key=feat)

    st.markdown("---")
    if st.button("Predict Benzene Concentration", type="primary", width='stretch'):
        input_df    = pd.DataFrame([inputs])
        scaled_input = scaler.transform(input_df)     # scaler fitted on train only

        pred = lr_pred.predict(scaled_input)[0]

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Predicted Benzene", f"{pred:.4f} (transformed)")
        col_r2.metric("Train-set Mean", f"{y_train.mean():.4f}")

        level = "Low" if pred < 5 else ("Moderate" if pred < 15 else "High")
        col_r3.metric("Risk Level", level)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Benzene Level (transformed scale)"},
            gauge={
                'axis': {'range': [float(y_train.min()), float(y_train.max())]},
                'bar':  {'color': "#1f77b4"},
                'steps': [
                    {'range': [float(y_train.min()), 5],  'color': '#d4edda'},
                    {'range': [5, 15],                    'color': '#fff3cd'},
                    {'range': [15, float(y_train.max())], 'color': '#f8d7da'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 15}
            }
        ))
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width='stretch')
