import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import warnings
warnings.filterwarnings("ignore")

# App title and icon
st.set_page_config(page_title="Dynamic Power Demand Forecasting", page_icon="⚡")
st.title("⚡ Dynamic Power Demand Forecasting")

# Sidebar inputs
st.sidebar.header("Model Configuration")
model_options = ["Prophet", "ARIMA", "LSTM", "Random Forest", "XGBoost"]
selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_options)
train_split = st.sidebar.slider("Training Data Percentage", min_value=50, max_value=95, value=80)

# Upload training and testing data
st.sidebar.header("Upload Data")
train_file = st.sidebar.file_uploader("Upload Training Data CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Testing Data CSV", type=["csv"])

# Function to calculate accuracy metrics
def calculate_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    accuracy = 100 - (mae / np.mean(true) * 100)
    return mae, rmse, accuracy

# Function to generate suggestions
def generate_suggestion(accuracy):
    if accuracy > 90:
        return "✅ Model is highly accurate."
    elif accuracy > 75:
        return "⚠️ Model is moderately accurate. Consider tuning hyperparameters."
    else:
        return "❌ Model is not accurate. Try different models or feature engineering."

# Main logic
if train_file is not None and test_file is not None:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Assume 'timestamp' and 'power_demand' columns exist
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    train_df.set_index('timestamp', inplace=True)
    test_df.set_index('timestamp', inplace=True)

    # Split training data
    split_index = int(len(train_df) * train_split / 100)
    train_data = train_df.iloc[:split_index]
    val_data = train_df.iloc[split_index:]

    # Forecasting
    forecast = []
    if selected_model == "Prophet":
        df_prophet = train_data.reset_index()[['timestamp', 'power_demand']]
        df_prophet.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df_prophet)
        future = test_df.reset_index()[['timestamp']]
        future.columns = ['ds']
        forecast_df = model.predict(future)
        forecast = forecast_df['yhat'].values

    elif selected_model == "ARIMA":
        model = ARIMA(train_data['power_demand'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test_df))

    elif selected_model == "LSTM":
        series = train_data['power_demand'].values
        n_input = 10
        generator = TimeseriesGenerator(series, series, length=n_input, batch_size=1)
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(generator, epochs=10, verbose=0)
        test_series = np.concatenate((series[-n_input:], test_df['power_demand'].values))
        predictions = []
        for i in range(len(test_df)):
            input_seq = test_series[i:i+n_input].reshape((1, n_input, 1))
            pred = lstm_model.predict(input_seq, verbose=0)
            predictions.append(pred[0][0])
        forecast = predictions

    elif selected_model == "Random Forest":
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data['power_demand'].values
        X_test = np.arange(len(train_data), len(train_data) + len(test_df)).reshape(-1, 1)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)

    elif selected_model == "XGBoost":
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data['power_demand'].values
        X_test = np.arange(len(train_data), len(train_data) + len(test_df)).reshape(-1, 1)
        model = XGBRegressor()
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)

    # Accuracy metrics
    mae, rmse, accuracy = calculate_metrics(test_df['power_demand'].values, forecast)
    st.sidebar.subheader("Model Accuracy")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")
    st.sidebar.write(f"Accuracy: {accuracy:.2f}%")
    st.sidebar.write(generate_suggestion(accuracy))

    # Forecast plot
    st.subheader("Forecast vs Actual")
    fig, ax = plt.subplots()
    ax.plot(test_df.index, test_df['power_demand'].values, label="Actual")
    ax.plot(test_df.index, forecast, label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # Summary
    st.markdown("### Summary")
    st.markdown(f"- Training Data Percentage: **{train_split}%**")
    st.markdown(f"- Selected Model: **{selected_model}**")
    st.markdown(f"- Accuracy: **{accuracy:.2f}%**")
