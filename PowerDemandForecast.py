
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb

st.set_page_config(page_title="Dynamic Power Demand Forecasting", page_icon="⚡", layout="wide")

st.title("⚡ Dynamic Power Demand Forecasting")

# Sidebar for metrics
st.sidebar.header("📊 Model Accuracy")
if 'accuracy' in st.session_state:
    st.sidebar.metric("Accuracy", f"{st.session_state['accuracy']:.2f}%")
    st.sidebar.write("Model is performing well." if st.session_state['accuracy'] > 80 else "Consider tuning hyperparameters.")
else:
    st.sidebar.write("Upload data to see accuracy.")

# Upload training data
st.subheader("Upload Training Data")
train_file = st.file_uploader("Upload CSV or Excel file for training", type=["csv", "xlsx"])
if train_file:
    if train_file.name.endswith('.csv'):
        df_train = pd.read_csv(train_file)
    else:
        df_train = pd.read_excel(train_file, engine='openpyxl')

    st.write("Training Data Sample:")
    st.dataframe(df_train.head())

    # Derived features
    df_train['hour'] = pd.to_datetime(df_train['time'], errors='coerce').dt.hour
    df_train['dayofweek'] = pd.to_datetime(df_train['date'], errors='coerce').dt.dayofweek

    # Train simple models
    features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
                'rain', 'snowfall', 'cloud_cover', 'wind_speed_100m', 'wind_speed_10m', 'hour', 'dayofweek']
    df_train = df_train.dropna(subset=features + ['demand'])
    X_train = df_train[features]
    y_train = df_train['demand']

    models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'XGBoost': xgb.XGBRegressor()
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_train)

    # Upload test data
    st.subheader("Upload Test Data")
    test_file = st.file_uploader("Upload CSV or Excel file for testing", type=["csv", "xlsx"], key="test")
    if test_file:
        if test_file.name.endswith('.csv'):
            df_test = pd.read_csv(test_file)
        else:
            df_test = pd.read_excel(test_file, engine='openpyxl')

        df_test['hour'] = pd.to_datetime(df_test['time'], errors='coerce').dt.hour
        df_test['dayofweek'] = pd.to_datetime(df_test['date'], errors='coerce').dt.dayofweek
        df_test = df_test.dropna(subset=features + ['demand'])
        X_test = df_test[features]
        y_test = df_test['demand']

        hybrid_pred = 0.25 * models['Random Forest'].predict(X_test) +                       0.25 * models['Linear Regression'].predict(X_test) +                       0.25 * models['SVR'].predict(X_test) +                       0.25 * models['XGBoost'].predict(X_test)

        mae = mean_absolute_error(y_test, hybrid_pred)
        rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
        accuracy = 100 - (mae / np.mean(y_test) * 100)
        st.session_state['accuracy'] = accuracy

        # Dropdown for state
        state_options = df_test['state'].unique()
        selected_state = st.selectbox("Select State", state_options)
        df_state = df_test[df_test['state'] == selected_state].copy()
        df_state['Forecast'] = hybrid_pred[:len(df_state)]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_state['date'], y=df_state['demand'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=df_state['date'], y=df_state['Forecast'], mode='lines+markers', name='Forecast'))
        fig.update_layout(title="Forecast vs Actual", xaxis_title="Date", yaxis_title="Demand")
        st.plotly_chart(fig, use_container_width=True)

        # Weightage info
        st.markdown("### 🔍 Model Weightages")
        st.write("Random Forest: 25%, Linear Regression: 25%, SVR: 25%, XGBoost: 25%")
        st.write("Input Variable Weightages: Equal")
        st.write("Derived Variable Weightages: Equal")

        # Sliders
        st.markdown("### 🎛️ Adjust Feature Weightages")
        for feature in features:
            st.slider(f"{feature} Weight", 0.0, 1.0, 1.0, 0.1)

        st.button("Optimize")
