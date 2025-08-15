import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor

st.set_page_config(page_title="Hybrid Power Demand Forecasting", layout="wide")

# Title
st.title("⚡ Multi-Model Hybrid Power Demand Forecasting App")

# File uploader
uploaded_train = st.file_uploader("Upload Training CSV", type=["csv"])
uploaded_test = st.file_uploader("Upload Testing CSV (Optional)", type=["csv"])

# Model selection
models_selected = st.multiselect(
    "Select Models for Hybrid Prediction",
    ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"],
    default=["Linear Regression", "Random Forest", "Gradient Boosting"]
)

# Weightage inputs
st.subheader("🔢 Model Weightages")
weights = {}
total_weight = 0
for model in models_selected:
    weight = st.number_input(f"Weight for {model}", min_value=0.0, max_value=1.0, value=1.0/len(models_selected))
    weights[model] = weight
    total_weight += weight

if total_weight == 0:
    st.error("Total weight cannot be zero!")

# Proceed if file uploaded
if uploaded_train:
    train_df = pd.read_csv(uploaded_train)
    st.write("### Training Data Preview", train_df.head())

    # Allow user to select target and features
    target_col = st.selectbox("Select Target Variable", train_df.columns)
    feature_cols = st.multiselect("Select Feature Columns", [c for c in train_df.columns if c != target_col], default=[c for c in train_df.columns if c != target_col])

    if feature_cols:
        X = train_df[feature_cols]
        y = train_df[target_col]

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store predictions
        val_predictions = pd.DataFrame(index=y_val.index)

        # Train models
        for model in models_selected:
            if model == "Linear Regression":
                m = LinearRegression()
            elif model == "Random Forest":
                m = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model == "Gradient Boosting":
                m = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model == "XGBoost":
                m = xgb.XGBRegressor(n_estimators=100, random_state=42)
            elif model == "LightGBM":
                m = LGBMRegressor(n_estimators=100, random_state=42)

            m.fit(X_train, y_train)
            preds = m.predict(X_val)
            val_predictions[model] = preds

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            st.write(f"**{model}** → RMSE: {rmse:.2f}, R²: {r2:.2f}")

        # Hybrid prediction
        val_predictions["Hybrid"] = sum(val_predictions[m] * weights[m] for m in models_selected) / total_weight

        hybrid_rmse = np.sqrt(mean_squared_error(y_val, val_predictions["Hybrid"]))
        hybrid_r2 = r2_score(y_val, val_predictions["Hybrid"])
        st.success(f"**Hybrid Model** → RMSE: {hybrid_rmse:.2f}, R²: {hybrid_r2:.2f}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_val, mode="lines+markers", name="Actual"))
        for model in models_selected + ["Hybrid"]:
            fig.add_trace(go.Scatter(y=val_predictions[model], mode="lines", name=model))
        st.plotly_chart(fig, use_container_width=True)

        # Testing
        if uploaded_test:
            test_df = pd.read_csv(uploaded_test)
            st.write("### Test Data Preview", test_df.head())

            test_preds = pd.DataFrame()
            for model in models_selected:
                if model == "Linear Regression":
                    m = LinearRegression()
                elif model == "Random Forest":
                    m = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model == "Gradient Boosting":
                    m = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model == "XGBoost":
                    m = xgb.XGBRegressor(n_estimators=100, random_state=42)
                elif model == "LightGBM":
                    m = LGBMRegressor(n_estimators=100, random_state=42)

                m.fit(X, y)
                test_preds[model] = m.predict(test_df[feature_cols])

            test_preds["Hybrid"] = sum(test_preds[m] * weights[m] for m in models_selected) / total_weight
            st.write("### Test Predictions", test_preds.head())
            csv = test_preds.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
