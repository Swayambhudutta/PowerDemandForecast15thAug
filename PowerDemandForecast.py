import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="⚡ Dynamic Power Demand Forecasting", layout="wide")
st.title("⚡ Dynamic Power Demand Forecasting")

# Sidebar for metrics
st.sidebar.header("📊 Model Accuracy")
accuracy_placeholder = st.sidebar.empty()
suggestion_placeholder = st.sidebar.empty()

# Upload training data
st.subheader("Upload Training Data")
train_file = st.file_uploader("Upload CSV or Excel file for training", type=["csv", "xlsx"])
if train_file:
    df_train = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file, engine='openpyxl')

    # Derived features
    df_train['hour'] = pd.to_datetime(df_train['time'], errors='coerce').dt.hour
    df_train['dayofweek'] = pd.to_datetime(df_train['date'], errors='coerce').dt.dayofweek

    features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
                'rain', 'snowfall', 'cloud_cover', 'wind_speed_100m', 'wind_speed_10m', 'hour', 'dayofweek']

    # Clean and validate training data
    df_train = df_train.dropna(subset=features + ['demand'])
    X_train = df_train[features].apply(pd.to_numeric, errors='coerce').dropna()
    y_train = pd.to_numeric(df_train.loc[X_train.index, 'demand'], errors='coerce')

    models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'XGBoost': XGBRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    st.success("✅ Training Completed")

    # Upload test data
    st.subheader("Upload Test Data")
    test_file = st.file_uploader("Upload CSV or Excel file for testing", type=["csv", "xlsx"], key="test")
    if test_file:
        df_test = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file, engine='openpyxl')
        df_test['hour'] = pd.to_datetime(df_test['time'], errors='coerce').dt.hour
        df_test['dayofweek'] = pd.to_datetime(df_test['date'], errors='coerce').dt.dayofweek
        df_test = df_test.dropna(subset=features + ['demand'])
        X_test = df_test[features].apply(pd.to_numeric, errors='coerce').dropna()
        y_test = pd.to_numeric(df_test.loc[X_test.index, 'demand'], errors='coerce')

        # Hybrid prediction
        hybrid_pred = sum(model.predict(X_test) * 0.25 for model in models.values())

        # Accuracy
        mae = mean_absolute_error(y_test, hybrid_pred)
        rmse = mean_squared_error(y_test, hybrid_pred, squared=False)
        accuracy = 100 - (mae / np.mean(y_test) * 100)
        accuracy_placeholder.metric("Accuracy", f"{accuracy:.2f}%")
        suggestion_placeholder.write("Model is accurate." if accuracy > 85 else "Consider tuning weights or adding more data.")

        # State selection
        selected_state = st.selectbox("Select State", df_test['state'].unique())
        df_state = df_test[df_test['state'] == selected_state].copy()
        df_state['Forecast'] = hybrid_pred[:len(df_state)]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_state['date'], y=df_state['demand'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=df_state['date'], y=df_state['Forecast'], mode='lines+markers', name='Forecast'))
        fig.update_layout(title="Forecast vs Actual", xaxis_title="Date", yaxis_title="Demand")
        st.plotly_chart(fig, use_container_width=True)

        # Sliders
        st.subheader("🔧 Adjust Feature Weightages")
        for feature in features:
            st.slider(f"{feature} Weight", 0.0, 1.0, 1.0, 0.1)

        st.button("Optimize")
