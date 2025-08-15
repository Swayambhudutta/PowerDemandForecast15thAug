import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Multi-Model Hybrid Forecast", layout="wide")

st.title("⚡ Multi-Model Hybrid Power Demand Forecasting App")

# ------------------------------
# File Upload
# ------------------------------
st.sidebar.header("Upload Your CSV")
train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

if train_file is not None:
    df_train = pd.read_csv(train_file)
    st.subheader("Training Data Preview")
    st.dataframe(df_train.head())

if test_file is not None:
    df_test = pd.read_csv(test_file)
    st.subheader("Test Data Preview")
    st.dataframe(df_test.head())

# ------------------------------
# User selects features & target
# ------------------------------
if train_file is not None:
    features = st.multiselect("Select Features (Independent Variables)", df_train.columns.tolist())
    target = st.selectbox("Select Target Variable (Dependent)", df_train.columns.tolist())

    if features and target:
        X = df_train[features]
        y = df_train[target]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # ------------------------------
        # Train Models
        # ------------------------------
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        lgb = LGBMRegressor(n_estimators=100, random_state=42)

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        lgb.fit(X_train, y_train)

        # Predictions
        rf_pred = rf.predict(X_val)
        xgb_pred = xgb.predict(X_val)
        lgb_pred = lgb.predict(X_val)

        # Hybrid prediction = average of models
        hybrid_pred = (rf_pred + xgb_pred + lgb_pred) / 3

        # ------------------------------
        # Model Performance
        # ------------------------------
        rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
        xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
        lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
        hybrid_rmse = np.sqrt(mean_squared_error(y_val, hybrid_pred))

        st.subheader("📊 Model RMSE Comparison")
        st.write({
            "RandomForest RMSE": rf_rmse,
            "XGBoost RMSE": xgb_rmse,
            "LightGBM RMSE": lgb_rmse,
            "Hybrid RMSE": hybrid_rmse
        })

        # ------------------------------
        # Feature Importance Visualization
        # ------------------------------
        rf_importances = rf.feature_importances_
        xgb_importances = xgb.feature_importances_
        lgb_importances = lgb.feature_importances_

        avg_importances = (rf_importances + xgb_importances + lgb_importances) / 3
        feature_df = pd.DataFrame({
            "Feature": features,
            "Avg Weight": avg_importances,
            "RF Weight": rf_importances,
            "XGB Weight": xgb_importances,
            "LGBM Weight": lgb_importances
        }).sort_values(by="Avg Weight", ascending=False)

        st.subheader("🔍 Feature Weightages Across Models")
        fig = px.bar(feature_df, x="Feature", y="Avg Weight", title="Average Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(feature_df)

        # ------------------------------
        # Predict on Test Data
        # ------------------------------
        if test_file is not None:
            X_test = df_test[features]
            rf_test_pred = rf.predict(X_test)
            xgb_test_pred = xgb.predict(X_test)
            lgb_test_pred = lgb.predict(X_test)

            hybrid_test_pred = (rf_test_pred + xgb_test_pred + lgb_test_pred) / 3
            df_test["Predicted_Target"] = hybrid_test_pred

            st.subheader("📈 Test Predictions")
            st.dataframe(df_test)

            csv_out = df_test.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv_out, "predictions.csv", "text/csv")
