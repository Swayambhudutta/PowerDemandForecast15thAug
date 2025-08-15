import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# App title
st.set_page_config(page_title="Multi-Model Hybrid Forecast", layout="wide")
st.title("🔮 Multi-Model Hybrid Power Demand Forecast")

st.markdown("""
This app trains a **multi-model hybrid regression** model to forecast demand using multiple features.
You can upload your own training and test CSV files, or use the sample data.
""")

# File upload
train_file = st.file_uploader("Upload Training CSV", type=["csv"])
test_file = st.file_uploader("Upload Test CSV", type=["csv"])

# Load data
if train_file is not None:
    df_train = pd.read_csv(train_file)
else:
    st.info("No training file uploaded. Using sample data...")
    df_train = pd.read_csv("sample_train.csv")

if test_file is not None:
    df_test = pd.read_csv(test_file)
else:
    df_test = pd.read_csv("sample_test.csv")

st.subheader("📊 Training Data Preview")
st.dataframe(df_train.head())

# Target variable selection
target_col = st.selectbox("Select Target Variable", df_train.columns)

# Feature selection
feature_cols = st.multiselect(
    "Select Feature Variables",
    [col for col in df_train.columns if col != target_col],
    default=[col for col in df_train.columns if col != target_col]
)

# Prepare data
X = df_train[feature_cols]
y = df_train[target_col]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

model_preds = {}
model_rmse = {}
model_r2 = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    model_preds[name] = preds
    model_rmse[name] = np.sqrt(mean_squared_error(y_val, preds))
    model_r2[name] = r2_score(y_val, preds)

# Hybrid model (average predictions)
hybrid_preds = np.mean(np.column_stack(list(model_preds.values())), axis=1)
model_preds["Hybrid Model"] = hybrid_preds
model_rmse["Hybrid Model"] = np.sqrt(mean_squared_error(y_val, hybrid_preds))
model_r2["Hybrid Model"] = r2_score(y_val, hybrid_preds)

# Model performance table
st.subheader("📈 Model Performance")
perf_df = pd.DataFrame({
    "RMSE": model_rmse,
    "R² Score": model_r2
}).sort_values(by="RMSE")
st.dataframe(perf_df)

# Feature importance visualization (from Random Forest)
st.subheader("📌 Feature Importances (Random Forest)")
rf_importances = models["Random Forest"].feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)

fig = go.Figure(data=[go.Bar(
    x=importance_df["Feature"],
    y=importance_df["Importance"],
    text=importance_df["Importance"].round(3),
    textposition='auto'
)])
fig.update_layout(title="Feature Importance Weights", xaxis_title="Features", yaxis_title="Weight")
st.plotly_chart(fig)

# Predict on test data
st.subheader("🔍 Test Data Predictions (Hybrid Model)")
X_test = df_test[feature_cols]
X_test_scaled = scaler.transform(X_test)
test_preds = np.mean([
    models["Linear Regression"].predict(X_test_scaled),
    models["Random Forest"].predict(X_test_scaled),
    models["Gradient Boosting"].predict(X_test_scaled)
], axis=0)
df_test["Predicted_" + target_col] = test_preds
st.dataframe(df_test)

# Download predictions
st.download_button(
    label="📥 Download Predictions",
    data=df_test.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)
