import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dynamic Power Demand Forecasting", page_icon="⚡")
st.title("⚡ Dynamic Power Demand Forecasting")

st.sidebar.header("Model Configuration")
model_options = ["ARIMA", "LSTM", "Random Forest", "XGBoost"]
selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_options)
train_split = st.sidebar.slider("Training Data Percentage", min_value=50, max_value=95, value=80)

st.sidebar.header("Upload Data")
train_file = st.sidebar.file_uploader("Upload Training Data CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Testing Data CSV", type=["csv"])

def calculate_metrics(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    accuracy = 100 - (mae / np.mean(true) * 100)
    return mae, rmse, accuracy

def generate_suggestion(accuracy):
    if accuracy > 90:
        return "✅ Model is highly accurate."
    elif accuracy > 75:
        return "⚠️ Model is moderately accurate. Consider tuning hyperparameters."
    else:
        return "❌ Model is not accurate. Try different models or feature engineering."

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    states = train_df['state'].unique().tolist()
    selected_state = st.selectbox("Select State for Forecasting", states)

    train_df = train_df[train_df['state'] == selected_state].sort_values('timestamp')
    test_df = test_df[test_df['state'] == selected_state].sort_values('timestamp')

    split_index = int(len(train_df) * train_split / 100)
    train_data = train_df.iloc[:split_index]

    forecast = []
    if selected_model == "ARIMA":
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
        lstm_model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
        lstm_model.fit(generator, epochs=10, verbose=0)
        test_series = np.concatenate((series[-n_input:], test_df['power_demand'].values))
        predictions = []
        for i in range(len(test_df)):
            input_seq = test_series[i:i+n_input].reshape((1, n_input, 1))
            pred = lstm_model.predict(input_seq, verbose=0)
            predictions.append(pred[0][0])
        forecast = predictions

    elif selected_model in ["Random Forest", "XGBoost"]:
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data['power_demand'].values
        X_test = np.arange(len(train_data), len(train_data) + len(test_df)).reshape(-1, 1)
        model = XGBRegressor(n_estimators=100 if selected_model == "Random Forest" else 200)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)

    mae, rmse, accuracy = calculate_metrics(test_df['power_demand'].values, forecast)
    st.sidebar.subheader("Model Accuracy")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")
    st.sidebar.write(f"Accuracy: {accuracy:.2f}%")
    st.sidebar.write(generate_suggestion(accuracy))

    st.subheader("Forecast vs Actual")
    plot_df = pd.DataFrame({
        'timestamp': test_df['timestamp'],
        'Actual': test_df['power_demand'],
        'Forecast': forecast
    })
    plot_df = plot_df.melt('timestamp', var_name='Type', value_name='Power Demand')
    chart = alt.Chart(plot_df).mark_line().encode(
        x='timestamp:T',
        y='Power Demand:Q',
        color='Type:N'
    ).properties(
        width=800,
        height=400,
        title=f"Forecast vs Actual for {selected_state}"
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Summary")
    st.markdown(f"- **State**: {selected_state}")
    st.markdown(f"- **Training Data Percentage**: {train_split}%")
    st.markdown(f"- **Selected Model**: {selected_model}")
    st.markdown(f"- **Accuracy**: {accuracy:.2f}%")
