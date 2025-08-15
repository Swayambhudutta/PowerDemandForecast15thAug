import streamlit as st
import pandas as pd
import numpy as np
import io

# ---------------------------
# Title and Sidebar
# ---------------------------
st.title("⚡ Power Demand Forecasting Tool")

st.sidebar.header("Upload and Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

weight_temp = st.sidebar.slider("Weight for Temperature", 0.0, 2.0, 1.0, 0.1)
weight_hour = st.sidebar.slider("Weight for Hour", 0.0, 2.0, 1.0, 0.1)
weight_day = st.sidebar.slider("Weight for Day of Week", 0.0, 2.0, 1.0, 0.1)
train_split_ratio = st.sidebar.slider("Training Data Split", 0.5, 0.9, 0.8, 0.05)

# ---------------------------
# Load Data
# ---------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # ---------------------------
    # Feature Engineering
    # ---------------------------
    if not {"Temperature", "Hour", "DayOfWeek", "Demand"}.issubset(df.columns):
        st.error("CSV must contain 'Temperature', 'Hour', 'DayOfWeek', 'Demand' columns.")
    else:
        # Apply weightages
        df["Temp_w"] = df["Temperature"] * weight_temp
        df["Hour_w"] = df["Hour"] * weight_hour
        df["Day_w"] = df["DayOfWeek"] * weight_day

        # Prepare features and labels
        X = df[["Temp_w", "Hour_w", "Day_w"]].values
        y = df["Demand"].values

        # ---------------------------
        # Train/Test Split
        # ---------------------------
        split_idx = int(len(X) * train_split_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # ---------------------------
        # Simple Linear Regression (No sklearn)
        # ---------------------------
        X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]  # Add bias term
        X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

        # Normal Equation: θ = (XᵀX)⁻¹ Xᵀy
        theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

        # Predictions
        y_pred_train = X_train_bias @ theta
        y_pred_test = X_test_bias @ theta

        # ---------------------------
        # Accuracy Metrics
        # ---------------------------
        mae = np.mean(np.abs(y_test - y_pred_test))
        rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - y_pred_test) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        st.subheader("📊 Model Accuracy Metrics")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # ---------------------------
        # Prediction Graph
        # ---------------------------
        st.subheader("📈 Prediction vs Actual (Test Data)")
        chart_data = pd.DataFrame({
            "Actual Demand": y_test,
            "Predicted Demand": y_pred_test
        }).reset_index(drop=True)
        st.line_chart(chart_data)

        # ---------------------------
        # Model Details
        # ---------------------------
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": ["Bias", "Temp_w", "Hour_w", "Day_w"],
            "Coefficient": theta
        })
        st.table(coef_df)
else:
    st.info("Please upload a CSV file to begin.")
