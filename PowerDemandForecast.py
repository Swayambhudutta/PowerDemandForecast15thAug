import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title="Multi-Model Hybrid Power Demand Forecast", layout="wide")

st.title("⚡ Multi-Model Hybrid Power Demand Forecast")

# Sidebar
st.sidebar.header("Upload Data & Configure Model")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Data")
    st.dataframe(df.head())

    target_col = st.sidebar.selectbox("Select Target Variable", df.columns)
    feature_cols = st.sidebar.multiselect(
        "Select Feature Variables", [col for col in df.columns if col != target_col]
    )

    if feature_cols:
        st.sidebar.subheader("Assign Weightages")
        weights = {}
        for col in feature_cols:
            weights[col] = st.sidebar.slider(f"Weight for {col}", 0.0, 5.0, 1.0, 0.1)

        # Apply weights
        for col in feature_cols:
            df[col] = df[col] * weights[col]

        # Train-test split
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models
        lr = LinearRegression()
        rf = RandomForestRegressor(random_state=42)
        gb = GradientBoostingRegressor(random_state=42)

        # Train models
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        # Predictions
        pred_lr = lr.predict(X_test)
        pred_rf = rf.predict(X_test)
        pred_gb = gb.predict(X_test)

        # Hybrid prediction (average of models)
        pred_hybrid = (pred_lr + pred_rf + pred_gb) / 3

        # Metrics
        r2 = r2_score(y_test, pred_hybrid)
        rmse = np.sqrt(mean_squared_error(y
