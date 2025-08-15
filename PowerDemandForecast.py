import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from typing import Dict, List, Tuple

# Classical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
from xgboost import XGBRegressor

# Prophet
from prophet import Prophet

# Deep Learning (Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Utilities
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dynamic Power Demand Forecasting", page_icon="⚡", layout="wide")

# ------------------------ UI Header ------------------------
c1, c2 = st.columns([0.1, 0.9])
with c1:
    st.markdown("<div style='font-size:36px;'>⚡</div>", unsafe_allow_html=True)
with c2:
    st.markdown("## Dynamic Power Demand Forecasting")

st.caption("Upload training & test data, select models and weights, tune feature weights, and generate a dynamic hybrid forecast for power demand.")

# ------------------------ Sidebar Controls ------------------------
st.sidebar.header("⚙️ Configuration")

# Model choice mode: single vs hybrid
mode = st.sidebar.selectbox("Prediction Mode", ["Hybrid (weighted)", "Single Model"])

# Model list
ALL_MODELS = [
    "ARIMA", "SARIMA", "SES", "AR", "MA",
    "Prophet",
    "LinearRegression", "RandomForest", "GradientBoosting", "XGBoost",
    "MLP",
    "SimpleRNN", "LSTM", "GRU",
    "VAR (multivariate target)"
]

if mode == "Single Model":
    chosen_models = [st.sidebar.selectbox("Choose a Model", ALL_MODELS)]
else:
    chosen_models = st.sidebar.multiselect(
        "Choose Models to Include in Hybrid",
        ALL_MODELS,
        default=["XGBoost", "LSTM", "SARIMA"]
    )

# Train size slider
train_ratio = st.sidebar.slider("Training share (%)", min_value=50, max_value=95, value=80, step=1)

# Placeholders for metrics
st.sidebar.subheader("📊 Performance")
metrics_box = st.sidebar.empty()
suggestion_box = st.sidebar.empty()

# ------------------------ Data Upload ------------------------
st.markdown("### 1) Upload Data")
c_up1, c_up2 = st.columns(2)
with c_up1:
    train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
with c_up2:
    test_file = st.file_uploader("Upload Testing CSV (optional, else we'll use holdout)", type=["csv"], key="test")

st.info("**Expected columns**: at minimum `timestamp`, `demand`. You may add exogenous variables (e.g., `temp`, `humidity`, `wind`, `is_holiday`, etc.)", icon="ℹ️")

def parse_df(f):
    df = pd.read_csv(f)
    if "timestamp" not in df.columns or "demand" not in df.columns:
        raise ValueError("CSV must include 'timestamp' and 'demand' columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

if train_file is not None:
    df_train_full = parse_df(train_file)
    if test_file is not None:
        df_test_full = parse_df(test_file)
    else:
        df_test_full = None
else:
    st.stop()

# ------------------------ Feature Engineering ------------------------
st.markdown("### 2) Feature Engineering & Variable Weights")

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["hour"] = x["timestamp"].dt.hour
    x["dayofweek"] = x["timestamp"].dt.dayofweek
    x["dayofyear"] = x["timestamp"].dt.dayofyear
    x["is_weekend"] = (x["dayofweek"] >= 5).astype(int)
    # cyclical encodings
    x["hour_sin"] = np.sin(2*np.pi*x["hour"]/24.0)
    x["hour_cos"] = np.cos(2*np.pi*x["hour"]/24.0)
    x["doy_sin"] = np.sin(2*np.pi*x["dayofyear"]/365.0)
    x["doy_cos"] = np.cos(2*np.pi*x["dayofyear"]/365.0)
    # lags & rolling features (derive from demand)
    for lag in [1, 2, 24, 48, 168]:
        x[f"lag_{lag}"] = x["demand"].shift(lag)
    x["roll_24_mean"] = x["demand"].rolling(24).mean()
    x["roll_24_std"] = x["demand"].rolling(24).std()
    x["roll_168_mean"] = x["demand"].rolling(168).mean()
    x["roll_168_std"] = x["demand"].rolling(168).std()
    return x

df_train_fe = add_time_features(df_train_full)
if df_test_full is not None:
    df_test_fe = add_time_features(df_test_full)
else:
    df_test_fe = None

# Determine candidate features (all except timestamp and demand)
feature_cols = [c for c in df_train_fe.columns if c not in ["timestamp", "demand"]]

# Dynamic variable weights
st.markdown("**Select and weight input variables**")
sel_features = st.multiselect("Features to include", feature_cols, default=feature_cols)
feat_weights = {}
if len(sel_features) > 0:
    for c in sel_features:
        feat_weights[c] = st.slider(f"Weight for {c}", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key=f"w_{c}")
    # normalize weights to sum to len(sel_features) -> effectively scaling but preserving relative preferences
    w_sum = sum(feat_weights.values()) if sum(feat_weights.values()) > 0 else 1.0
    for k in feat_weights:
        feat_weights[k] = feat_weights[k] / w_sum
else:
    st.warning("Please select at least one feature.", icon="⚠️")
    st.stop()

def apply_weights(df: pd.DataFrame, cols: List[str], weights: Dict[str, float]) -> pd.DataFrame:
    x = df.copy()
    for c in cols:
        x[c] = x[c] * weights.get(c, 1.0)
    return x

df_train_fe = apply_weights(df_train_fe, sel_features, feat_weights)
if df_test_fe is not None:
    df_test_fe = apply_weights(df_test_fe, sel_features, feat_weights)

# Train/holdout split (if no test file provided)
if df_test_fe is None:
    split_idx = int(len(df_train_fe) * (train_ratio/100.0))
    df_tr = df_train_fe.iloc[:split_idx].copy()
    df_val = df_train_fe.iloc[split_idx:].copy()
else:
    df_tr = df_train_fe.copy()
    df_val = df_test_fe.copy()

# Drop rows with NaNs caused by lags/rolling in training
df_tr = df_tr.dropna().reset_index(drop=True)
df_val = df_val.dropna().reset_index(drop=True)

X_train = df_tr[sel_features].values
y_train = df_tr["demand"].values
X_val = df_val[sel_features].values
y_val = df_val["demand"].values

# ------------------------ Model Builders ------------------------
def predict_arima_like(y_series: pd.Series, steps:int, order:Tuple[int,int,int]=(5,0,0), seasonal:Tuple[int,int,int,int]=None):
    if seasonal is None:
        m = ARIMA(y_series, order=order).fit()
        fc = m.forecast(steps=steps)
    else:
        m = SARIMAX(y_series, order=(order[0], order[1], order[2]), seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps)
    return np.array(fc)

def predict_ses(y_series: pd.Series, steps:int, seasonal_periods:int=24):
    model = ExponentialSmoothing(y_series, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
    fc = model.forecast(steps)
    return np.array(fc)

def predict_prophet(train_df: pd.DataFrame, future_timestamps: pd.Series):
    df = train_df[["timestamp","demand"]].rename(columns={"timestamp":"ds", "demand":"y"})
    m = Prophet()
    m.fit(df)
    future = pd.DataFrame({"ds": future_timestamps})
    fc = m.predict(future)
    return fc["yhat"].values

def make_sklearn(model, X_tr, y_tr, X_te):
    mdl = model
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te)

def make_xgb(X_tr, y_tr, X_te):
    mdl = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42
    )
    mdl.fit(X_tr, y_tr, verbose=False)
    return mdl.predict(X_te)

def make_mlp(X_tr, y_tr, X_te):
    mdl = MLPRegressor(hidden_layer_sizes=(128,64), activation="relu", random_state=42, max_iter=500)
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te)

def make_keras_rnn(X_tr, y_tr, X_te, kind="LSTM", lookback=24):
    # Build supervised sequences
    def build_seq(X, y, lb):
        seq_X, seq_y = [], []
        for i in range(lb, len(X)):
            seq_X.append(X[i-lb:i, :])
            seq_y.append(y[i])
        return np.array(seq_X), np.array(seq_y)

    scaler = StandardScaler()
    X_all = np.vstack([X_tr, X_te])
    X_all_scaled = scaler.fit_transform(X_all)
    # Split back
    X_tr_s = X_all_scaled[:len(X_tr)]
    X_te_s = X_all_scaled[len(X_tr):]

    X_tr_seq, y_tr_seq = build_seq(X_tr_s, y_tr, lookback)
    # For test, we need sequences too; prepend tail of train to construct initial context
    context = X_tr_s[-lookback:]
    X_te_seq = []
    # step through test points one by one with sliding window on features only
    buffer = list(context)
    for i in range(len(X_te_s)):
        X_te_seq.append(np.array(buffer[-lookback:]))
        buffer.append(X_te_s[i])
    X_te_seq = np.array(X_te_seq)

    model = Sequential()
    if kind == "LSTM":
        model.add(LSTM(64, input_shape=(lookback, X_tr.shape[1])))
    elif kind == "GRU":
        model.add(GRU(64, input_shape=(lookback, X_tr.shape[1])))
    else:
        model.add(SimpleRNN(64, input_shape=(lookback, X_tr.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X_tr_seq, y_tr_seq, epochs=20, batch_size=64, verbose=0, callbacks=[es])
    preds = model.predict(X_te_seq, verbose=0).reshape(-1)
    return preds

# ------------------------ Run Models ------------------------
st.markdown("### 3) Model Weights & Forecasting")

if mode == "Hybrid (weighted)":
    # sliders for each chosen model
    st.markdown("**Choose model weights**")
    model_weights = {}
    cols = st.columns(min(4, len(chosen_models)) if len(chosen_models) > 0 else 1)
    for idx, m in enumerate(chosen_models):
        with cols[idx % len(cols)]:
            model_weights[m] = st.slider(f"{m} weight", 0.0, 1.0, 0.3 if idx==0 else 0.2, 0.05)
    wsum = sum(model_weights.values()) if sum(model_weights.values())>0 else 1.0
    for k in model_weights:
        model_weights[k] = model_weights[k] / wsum
else:
    model_weights = {chosen_models[0]: 1.0}

# Prepare holder for individual model predictions
preds_by_model: Dict[str, np.ndarray] = {}

# Forecast horizon equals len of validation set
h = len(df_val)

# Time index for validation
ts_val = df_val["timestamp"].values

# Run models as requested
y_series_train = df_tr["demand"]

for m in model_weights.keys():
    try:
        if m == "ARIMA":
            preds = predict_arima_like(y_series_train, steps=h, order=(3,0,2))
        elif m == "SARIMA":
            # assuming hourly seasonality 24
            preds = predict_arima_like(y_series_train, steps=h, order=(2,0,2), seasonal=(1,0,1,24))
        elif m == "SES":
            preds = predict_ses(y_series_train, steps=h, seasonal_periods=24)
        elif m == "AR":
            preds = predict_arima_like(y_series_train, steps=h, order=(5,0,0))
        elif m == "MA":
            preds = predict_arima_like(y_series_train, steps=h, order=(0,0,5))
        elif m == "Prophet":
            preds = predict_prophet(df_tr, df_val["timestamp"])
        elif m == "LinearRegression":
            preds = make_sklearn(LinearRegression(), X_train, y_train, X_val)
        elif m == "RandomForest":
            preds = make_sklearn(RandomForestRegressor(n_estimators=300, random_state=42), X_train, y_train, X_val)
        elif m == "GradientBoosting":
            preds = make_sklearn(GradientBoostingRegressor(random_state=42), X_train, y_train, X_val)
        elif m == "XGBoost":
            preds = make_xgb(X_train, y_train, X_val)
        elif m == "MLP":
            preds = make_mlp(X_train, y_train, X_val)
        elif m == "SimpleRNN":
            preds = make_keras_rnn(X_train, y_train, X_val, kind="RNN", lookback=24)
        elif m == "LSTM":
            preds = make_keras_rnn(X_train, y_train, X_val, kind="LSTM", lookback=24)
        elif m == "GRU":
            preds = make_keras_rnn(X_train, y_train, X_val, kind="GRU", lookback=24)
        elif m == "VAR (multivariate target)":
            # Requires multi-endogenous setup; here we attempt VAR on demand + selected lag features if available
            var_df = df_tr[["demand"] + [c for c in sel_features if c.startswith("lag_")][:2]].dropna()
            if var_df.shape[1] >= 2 and len(var_df) > 50:
                model = VAR(var_df)
                res = model.fit(maxlags=12, ic="aic")
                # Forecast steps ahead based on last 'k_ar' observations
                lag_order = res.k_ar
                fc = res.forecast(var_df.values[-lag_order:], steps=h)
                preds = fc[:, 0]  # demand is first column
            else:
                preds = predict_arima_like(y_series_train, steps=h, order=(3,0,2))
        else:
            preds = np.full(h, np.nan)
        preds_by_model[m] = preds
    except Exception as e:
        preds_by_model[m] = np.full(h, np.nan)
        st.warning(f"{m} failed: {e}")

# Hybrid combine
hybrid_pred = np.zeros(h)
for m, w in model_weights.items():
    p = preds_by_model.get(m, np.full(h, np.nan))
    if np.isnan(p).any():
        continue
    hybrid_pred += w * p

# If in Single Model mode, use its predictions directly
if mode == "Single Model":
    only = list(model_weights.keys())[0]
    hybrid_pred = preds_by_model.get(only, hybrid_pred)

# ------------------------ Metrics ------------------------
def compute_metrics(y_true, y_hat):
    mae = mean_absolute_error(y_true, y_hat)
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mape = np.mean(np.abs((y_true - y_hat) / np.maximum(1e-6, np.abs(y_true)))) * 100
    r2 = r2_score(y_true, y_hat)
    return mae, rmse, mape, r2

mae, rmse, mape, r2 = compute_metrics(y_val, hybrid_pred)
acc_pct = max(0.0, 100.0 - mape)

with metrics_box:
    st.sidebar.metric("MAE", f"{mae:,.2f}")
    st.sidebar.metric("RMSE", f"{rmse:,.2f}")
    st.sidebar.metric("MAPE", f"{mape:,.2f}%")
    st.sidebar.metric("Accuracy (100 - MAPE)", f"{acc_pct:,.2f}%")

# Simple suggestion logic
suggestions = []
if mape > 15:
    suggestions.append("High MAPE: try increasing training share or add more informative features (e.g., weather, calendar effects, lags).")
if r2 < 0.7:
    suggestions.append("Low R²: try non-linear models (XGBoost/RandomForest) or deep models (LSTM/GRU).")
if "SARIMA" not in model_weights and "Prophet" not in model_weights:
    suggestions.append("Add a seasonal model (SARIMA/Prophet) for hourly/weekly seasonality.")
if len(sel_features) < 5:
    suggestions.append("Include more derived variables (lags, rolling stats, cyclical encodings).")
if not suggestions:
    suggestions.append("Model selection and features look reasonable. Consider fine-tuning weights for marginal gains.")

with suggestion_box:
    st.sidebar.write("🧠 **Suggestions**")
    for s in suggestions:
        st.sidebar.write(f"- {s}")

# ------------------------ Plot ------------------------
st.markdown("### 4) Forecast vs Actual")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_val["timestamp"], y=y_val, name="Actual", mode="lines"))
fig.add_trace(go.Scatter(x=df_val["timestamp"], y=hybrid_pred, name="Forecast", mode="lines"))
st.plotly_chart(fig, use_container_width=True)

# ------------------------ Notes ------------------------
st.markdown("##### Notes")
st.write(f"- **Training share**: {train_ratio}%")
st.write("- **Model weights**:")
mw_table = pd.DataFrame({"Model": list(model_weights.keys()), "Weight": [round(model_weights[k],3) for k in model_weights]})
st.dataframe(mw_table, use_container_width=True)

st.write("- **Feature (variable) weights**:")
fw_table = pd.DataFrame({"Feature": list(feat_weights.keys()), "Weight": [round(feat_weights[k],3) for k in feat_weights]})
st.dataframe(fw_table, use_container_width=True)

# ------------------------ Details ------------------------
st.markdown("### 5) Inputs and Hybrid Configuration")
c_a, c_b = st.columns(2)
with c_a:
    st.markdown("**Input Variables Used**")
    st.write(", ".join(sel_features))
with c_b:
    st.markdown("**Models Included**")
    st.write(", ".join(model_weights.keys()))

st.success("Done! Adjust sliders (training share, model weights, feature weights) and re-run to update the forecast dynamically.")
