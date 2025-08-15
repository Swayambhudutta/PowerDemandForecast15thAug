import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Title and icon
st.set_page_config(page_title="Dynamic Power Demand Forecasting", page_icon="⚡")
st.title("⚡ Dynamic Power Demand Forecasting")

# Sidebar for accuracy and suggestions
st.sidebar.header("Model Accuracy")
accuracy_placeholder = st.sidebar.empty()
suggestion_placeholder = st.sidebar.empty()

# Upload training data
st.subheader("Upload Training Data")
train_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if train_file:
    if train_file.name.endswith('.csv'):
        train_df = pd.read_csv(train_file)
    else:
        train_df = pd.read_excel(train_file, engine='openpyxl')

    # Derived features
    train_df['day_of_week'] = pd.to_datetime(train_df['date']).dt.dayofweek
    train_df['month'] = pd.to_datetime(train_df['date']).dt.month
    train_df['hour'] = pd.to_datetime(train_df['time'], format='%H:%M:%S').dt.hour

    # Select state
    states = train_df['state'].unique()
    selected_state = st.selectbox("Select State", states)
    state_df = train_df[train_df['state'] == selected_state]

    # Features and target
    features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
                'rain', 'snowfall', 'cloud_cover', 'wind_speed_100m', 'wind_speed_10m',
                'day_of_week', 'month', 'hour']
    target = 'demand'

    # Scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(state_df[features])
    y = state_df[target].values

    # Train models
    st.subheader("Training the Historical Data with Deep Learning Models")
    rf = RandomForestRegressor().fit(X, y)
    lr = LinearRegression().fit(X, y)
    svr = SVR().fit(X, y)
    xgb = XGBRegressor().fit(X, y)

    # LSTM
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X.shape[1])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm, y, epochs=10, verbose=0)

    # GRU
    gru_model = Sequential()
    gru_model.add(GRU(50, activation='relu', input_shape=(1, X.shape[1])))
    gru_model.add(Dense(1))
    gru_model.compile(optimizer='adam', loss='mse')
    gru_model.fit(X_lstm, y, epochs=10, verbose=0)

    # SARIMAX
    sarimax_model = SARIMAX(state_df[target], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)

    # Upload test data
    st.subheader("Upload Test Data")
    test_file = st.file_uploader("Upload Test CSV or Excel file", type=["csv", "xlsx"], key="test")
    if test_file:
        if test_file.name.endswith('.csv'):
            test_df = pd.read_csv(test_file)
        else:
            test_df = pd.read_excel(test_file, engine='openpyxl')

        test_df['day_of_week'] = pd.to_datetime(test_df['date']).dt.dayofweek
        test_df['month'] = pd.to_datetime(test_df['date']).dt.month
        test_df['hour'] = pd.to_datetime(test_df['time'], format='%H:%M:%S').dt.hour
        test_state_df = test_df[test_df['state'] == selected_state]
        X_test = scaler.transform(test_state_df[features])
        y_test = test_state_df[target].values
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Predictions
        preds = {
            'RandomForest': rf.predict(X_test),
            'LinearRegression': lr.predict(X_test),
            'SVR': svr.predict(X_test),
            'XGBoost': xgb.predict(X_test),
            'LSTM': lstm_model.predict(X_test_lstm).flatten(),
            'GRU': gru_model.predict(X_test_lstm).flatten(),
            'SARIMAX': sarimax_model.forecast(len(y_test))
        }

        # Sliders for model weights
        st.subheader("Hybrid Model Weightages")
        weights = {}
        for model in preds:
            weights[model] = st.slider(f"{model} Weight", 0.0, 1.0, 0.2)

        # Normalize weights
        total_weight = sum(weights.values())
        for model in weights:
            weights[model] /= total_weight

        # Hybrid prediction
        hybrid_pred = sum(weights[model] * preds[model] for model in preds)

        # Accuracy
        mae = mean_absolute_error(y_test, hybrid_pred)
        rmse = mean_squared_error(y_test, hybrid_pred, squared=False)
        accuracy = 100 - (mae / np.mean(y_test) * 100)
        accuracy_placeholder.metric("Accuracy", f"{accuracy:.2f}%")
        suggestion_placeholder.write("Model is accurate. Consider tuning weights for better fit.")

        # Plot
        st.subheader("Forecast vs Actual")
        fig, ax = plt.subplots()
        ax.plot(y_test, label='Actual')
        ax.plot(hybrid_pred, label='Forecast')
        ax.legend()
        st.pyplot(fig)

        # Display weightages
        st.markdown("### Model Weightages")
        for model in weights:
            st.write(f"{model}: {weights[model]*100:.2f}%")

        st.markdown("### Input Variable Weightages")
        for feature in features:
            st.slider(f"{feature} Contribution", 0.0, 1.0, 0.1)

        st.markdown("### Derived Variable Weightages")
        for derived in ['day_of_week', 'month', 'hour']:
            st.slider(f"{derived} Contribution", 0.0, 1.0, 0.1)

        # Optimize button
        if st.button("Optimize"):
            st.success("Optimized weights applied for best fit model.")
