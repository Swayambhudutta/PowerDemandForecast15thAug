import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------
# Simulated Data (Replace with your CSV if needed)
# ------------------------------
def generate_sample_data():
    np.random.seed(42)
    days = np.arange(1, 31)  # 30 days
    demand = 200 + 5 * days + np.random.normal(0, 10, size=len(days))
    return pd.DataFrame({"Day": days, "Demand": demand})

# ------------------------------
# Simple Linear Regression (No sklearn)
# ------------------------------
def simple_linear_regression(x, y):
    # Formula: y = m*x + c
    m = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    c = np.mean(y) - m * np.mean(x)
    return m, c

# ------------------------------
# Streamlit App
# ------------------------------
st.title("⚡ Power Demand Forecast")
st.write("Forecast daily power demand using a simple linear regression (no sklearn).")

# Load Data
df = generate_sample_data()
st.subheader("Sample Power Demand Data")
st.dataframe(df)

# Train Model
X = df["Day"].values
y = df["Demand"].values
m, c = simple_linear_regression(X, y)

# Predict for next 7 days
future_days = np.arange(max(X) + 1, max(X) + 8)
future_demand = m * future_days + c
forecast_df = pd.DataFrame({"Day": future_days, "Predicted Demand": future_demand})

# Show Results
st.subheader("Forecasted Demand for Next 7 Days")
st.dataframe(forecast_df)

# Chart (Streamlit built-in, no matplotlib/plotly)
st.subheader("Demand Forecast Chart")
chart_df = pd.concat([
    df.rename(columns={"Demand": "Value"}).assign(Type="Actual"),
    forecast_df.rename(columns={"Predicted Demand": "Value"}).assign(Type="Forecast")
])
st.line_chart(chart_df.pivot(index="Day", columns="Type", values="Value"))

st.success("✅ Forecast completed without sklearn/matplotlib/plotly.")
