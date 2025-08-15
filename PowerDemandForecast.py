
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(page_title="Dynamic Power Demand Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ Dynamic Power Demand Forecasting")

# Sidebar
st.sidebar.header("Model Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
split_ratio = st.sidebar.slider("Train-Test Split (%)", 50, 90, 80)
model_options = ["Linear Regression", "Random Forest", "SVR", "XGBoost", "SARIMAX", "LSTM", "GRU", "Hybrid"]
selected_model = st.sidebar.selectbox("Select Forecasting Model", model_options)

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    states = df['state'].unique()
    selected_state = st.selectbox("Select State", states)
    df_state = df[df['state'] == selected_state]

    # Robust datetime parsing
    df_state['datetime'] = pd.to_datetime(df_state['date'].astype(str) + ' ' + df_state['time'].astype(str), errors='coerce')
    df_state = df_state.dropna(subset=['datetime'])

    # Derived features
    df_state['hour'] = df_state['datetime'].dt.hour
    df_state['day_of_week'] = df_state['datetime'].dt.dayofweek
    df_state['month'] = df_state['datetime'].dt.month

    input_features = ['temperature_2m', 'weather_code', 'relative_humidity_2m', 'dew_point_2m',
                      'apparent_temperature', 'rain', 'snowfall', 'cloud_cover',
                      'wind_speed_100m', 'wind_speed_10m', 'hour', 'day_of_week', 'month']
    target = 'demand'

    # Normalize input features
    scaler = MinMaxScaler()
    df_state[input_features] = scaler.fit_transform(df_state[input_features])

    # Initialize feature weights
    if 'feature_weights' not in st.session_state:
        st.session_state.feature_weights = {f: 100 for f in input_features}

    # Apply feature weights
    for f in input_features:
        df_state[f] *= st.session_state.feature_weights[f] / 100.0

    X = df_state[input_features]
    y = df_state[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio/100, shuffle=False)

    st.subheader("Training the Historical Data with Deep Learning Models")
    y_pred = None
    weights = {}

    if selected_model == "Linear Regression":
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif selected_model == "Random Forest":
        model = RandomForestRegressor().fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif selected_model == "SVR":
        model = SVR().fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif selected_model == "XGBoost":
        model = xgb.XGBRegressor().fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif selected_model == "SARIMAX":
        model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24)).fit(disp=False)
        y_pred = model.forecast(steps=len(y_test))

    elif selected_model in ["LSTM", "GRU"]:
        generator = TimeseriesGenerator(X_train.values, y_train.values, length=24, batch_size=1)
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(24, X_train.shape[1])) if selected_model == "LSTM"
                  else GRU(50, activation='relu', input_shape=(24, X_train.shape[1])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(generator, epochs=5)
        test_gen = TimeseriesGenerator(X_test.values, y_test.values, length=24, batch_size=1)
        y_pred = model.predict(test_gen).flatten()

    elif selected_model == "Hybrid":
        weights = {"XGBoost": 0.3, "LSTM": 0.4, "GRU": 0.3}
        pred_xgb = xgb.XGBRegressor().fit(X_train, y_train).predict(X_test)

        def train_dl_model(model_type):
            gen = TimeseriesGenerator(X_train.values, y_train.values, length=24, batch_size=1)
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(24, X_train.shape[1])) if model_type == "LSTM"
                      else GRU(50, activation='relu', input_shape=(24, X_train.shape[1])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(gen, epochs=5)
            test_gen = TimeseriesGenerator(X_test.values, y_test.values, length=24, batch_size=1)
            return model.predict(test_gen).flatten()

        pred_lstm = train_dl_model("LSTM")
        pred_gru = train_dl_model("GRU")
        y_pred = weights["XGBoost"] * pred_xgb[:len(pred_lstm)] + weights["LSTM"] * pred_lstm + weights["GRU"] * pred_gru

    # Accuracy
    mae = mean_absolute_error(y_test[:len(y_pred)], y_pred)
    mse = mean_squared_error(y_test[:len(y_pred)], y_pred)
    r2 = r2_score(y_test[:len(y_pred)], y_pred)
    accuracy = 100 - (mae / y_test.mean()) * 100

    st.sidebar.subheader("Statistical Measurement")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"MSE: {mse:.2f}")
    st.sidebar.write(f"R²: {r2:.2f}")
    st.sidebar.write(f"Accuracy: {accuracy:.2f}%")

    if accuracy > 85:
        st.sidebar.success("Model is highly accurate.")
    elif accuracy > 70:
        st.sidebar.warning("Model is moderately accurate.")
    else:
        st.sidebar.error("Model accuracy is low.")

    # Line plot
    st.subheader("Forecasting vs Actual")
    fig, ax = plt.subplots()
    ax.plot(y_test[:len(y_pred)].values, label="Actual", linestyle='-', marker='')
    ax.plot(y_pred, label="Forecast", linestyle='-', marker='')
    ax.legend()
    st.pyplot(fig)

    # Feature sliders and optimize button below the graph
    st.subheader("Adjust Feature Weightages")
    new_weights = {}
    for f in input_features:
        new_weights[f] = st.slider(f"{f} (%)", 0, 100, st.session_state.feature_weights[f], key=f, step=1)

    if st.button("Optimize"):
        total = sum(np.random.rand(len(input_features)))
        optimized_weights = {f: int((np.random.rand() / total) * 100) for f in input_features}
        st.session_state.feature_weights = optimized_weights
        st.experimental_rerun()

    # Notes
    st.markdown(f"""
    **Notes:**
    - Train: {split_ratio}%, Test: {100 - split_ratio}%
    - Model Weights (Hybrid): {weights if selected_model == "Hybrid" else "N/A"}
    - Input Variable Weights: {st.session_state.feature_weights}
    - Derived Variables: hour, day_of_week, month
    """)
