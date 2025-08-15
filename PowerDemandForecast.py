# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("Multi-Model Hybrid Forecasting App")

# File uploaders
train_file = st.file_uploader("Upload Training CSV", type="csv")
test_file = st.file_uploader("Upload Testing CSV", type="csv")

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    st.subheader("Training Data")
    st.write(train_df.head())

    st.subheader("Testing Data")
    st.write(test_df.head())

    # Select target and features
    target_var = st.selectbox("Select Target Variable", train_df.columns)
    feature_vars = st.multiselect("Select Feature Variables", [col for col in train_df.columns if col != target_var], default=list(train_df.columns[:-1]))

    # Weightages input
    st.subheader("Assign Weightages to Models (Total should be 1.0)")
    w_lr = st.number_input("Weightage for Linear Regression", min_value=0.0, max_value=1.0, value=0.33)
    w_rf = st.number_input("Weightage for Random Forest", min_value=0.0, max_value=1.0, value=0.33)
    w_gb = st.number_input("Weightage for Gradient Boosting", min_value=0.0, max_value=1.0, value=0.34)

    if abs((w_lr + w_rf + w_gb) - 1.0) > 0.001:
        st.warning("The sum of weightages must be 1.0")
    else:
        # Train models
        X_train = train_df[feature_vars]
        y_train = train_df[target_var]
        X_test = test_df[feature_vars]
        y_test = test_df[target_var]

        model_lr = LinearRegression()
        model_rf = RandomForestRegressor(random_state=42)
        model_gb = GradientBoostingRegressor(random_state=42)

        model_lr.fit(X_train, y_train)
        model_rf.fit(X_train, y_train)
        model_gb.fit(X_train, y_train)

        pred_lr = model_lr.predict(X_test)
        pred_rf = model_rf.predict(X_test)
        pred_gb = model_gb.predict(X_test)

        # Hybrid prediction
        final_pred = (w_lr * pred_lr) + (w_rf * pred_rf) + (w_gb * pred_gb)

        # Metrics
        mae = mean_absolute_error(y_test, final_pred)
        rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        r2 = r2_score(y_test, final_pred)

        st.subheader("Model Performance (Hybrid)")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"R² Score: {r2:.4f}")

        # Visualization with Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test.values, label="Actual", marker='o')
        ax.plot(final_pred, label="Predicted", marker='x')
        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Index")
        ax.set_ylabel(target_var)
        ax.legend()
        st.pyplot(fig)

        # Show weightages visually
        fig2, ax2 = plt.subplots()
        ax2.bar(["Linear Regression", "Random Forest", "Gradient Boosting"], [w_lr, w_rf, w_gb], color=['blue', 'green', 'orange'])
        ax2.set_title("Model Weightages")
        st.pyplot(fig2)
