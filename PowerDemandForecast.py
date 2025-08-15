# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Multi-Model Hybrid Power Demand Forecasting")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Select features and target
    target_col = st.selectbox("Select Target Variable (Demand)", df.columns)
    feature_cols = st.multiselect("Select Feature Variables", [c for c in df.columns if c != target_col])

    if feature_cols:
        # Assign weights for each feature
        st.subheader("Assign Weights to Features")
        weights = {}
        for col in feature_cols:
            weights[col] = st.slider(f"Weight for {col}", 0.0, 1.0, 0.5)

        # Weighted feature transformation
        X = df[feature_cols].copy()
        for col in feature_cols:
            X[col] = X[col] * weights[col]
        y = df[target_col]

        # Train-Test split (manual to avoid train_test_split dependency)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Model 1: Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # Model 2: Random Forest
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Hybrid prediction (average of both models)
        y_pred_hybrid = (y_pred_lr + y_pred_rf) / 2

        # Metrics
        st.subheader("Model Performance")
        st.write(f"Linear Regression R²: {r2_score(y_test, y_pred_lr):.3f}")
        st.write(f"Random Forest R²: {r2_score(y_test, y_pred_rf):.3f}")
        st.write(f"Hybrid Model R²: {r2_score(y_test, y_pred_hybrid):.3f}")

        # Combine results for plotting
        results_df = pd.DataFrame({
            "Actual": y_test.values,
            "Linear Regression": y_pred_lr,
            "Random Forest": y_pred_rf,
            "Hybrid": y_pred_hybrid
        })

        st.subheader("Forecast vs Actual")
        st.line_chart(results_df)

        # Show feature weights
        weight_df = pd.DataFrame(list(weights.items()), columns=["Feature", "Weight"])
        st.subheader("Feature Weights")
        st.bar_chart(weight_df.set_index("Feature"))
else:
    st.info("Please upload a CSV file to proceed.")
