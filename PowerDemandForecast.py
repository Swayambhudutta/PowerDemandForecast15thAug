import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

# Sidebar - weightages
st.sidebar.header("Model Weightages")
weight_feature1 = st.sidebar.slider("Weight for Feature 1", 0.0, 1.0, 0.5)
weight_feature2 = st.sidebar.slider("Weight for Feature 2", 0.0, 1.0, 0.5)

st.title("Power Demand Forecast with Prediction Graph")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.write(data.head())

    # Assume first column is Date/Time, last column is target
    features = data.iloc[:, 1:-1]
    target = data.iloc[:, -1]

    # Apply weightages (if at least 2 features)
    if features.shape[1] >= 2:
        features.iloc[:, 0] *= weight_feature1
        features.iloc[:, 1] *= weight_feature2

    # Train-test split
    split = int(len(data) * 0.8)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Accuracy metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    st.write("### Statistical Accuracy")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Prediction vs Actual chart
    chart_data = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    chart_data["Index"] = chart_data.index

    chart = alt.Chart(chart_data).transform_fold(
        ["Actual", "Predicted"],
        as_=["Type", "Value"]
    ).mark_line().encode(
        x="Index",
        y="Value:Q",
        color="Type:N"
    ).properties(
