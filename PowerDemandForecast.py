import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import altair as alt

# ----------------------
# Title
# ----------------------
st.title("⚡ Power Demand Forecasting App")

# ----------------------
# File Upload
# ----------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("No file uploaded. Using sample dataset.")
    np.random.seed(42)
    df = pd.DataFrame({
        "Temperature": np.random.uniform(15, 35, 100),
        "Humidity": np.random.uniform(40, 80, 100),
        "WindSpeed": np.random.uniform(0, 15, 100),
        "Demand": np.random.uniform(200, 500, 100)
    })

st.write("### Dataset Preview")
st.dataframe(df.head())

# ----------------------
# Feature & Target Selection
# ----------------------
features = st.multiselect("Select features for prediction", options=df.columns.tolist(), default=df.columns[:-1])
target = st.selectbox("Select target variable", options=df.columns.tolist(), index=len(df.columns) - 1)

# ----------------------
# Model Training
# ----------------------
if features and target:
    X = df[features]
    y = df[target]

    test_size = st.sidebar.slider("Test set size (%)", 10, 50, 20) / 100
    random_state = st.sidebar.number_input("Random state (for reproducibility)", value=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # ----------------------
    # Metrics
    # ----------------------
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    st.subheader("📊 Model Performance")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**MAE:** {mae:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")

    # ----------------------
    # Actual vs Predicted Graph
    # ----------------------
    st.subheader("📈 Actual vs Predicted")

    df_plot = pd.DataFrame({
        "Time": np.arange(len(y_test)),
        "Actual": y_test.values,
        "Predicted": predictions
    })

    chart = alt.Chart(df_plot).transform_fold(
        ['Actual', 'Predicted'],
        as_=['Type', 'Value']
    ).mark_line(point=True).encode(
        x='Time:Q',
        y='Value:Q',
        color='Type:N'
    ).properties(
        width=700,
        height=400,
        title='Actual vs Predicted Power Demand'
    )

    st.altair_chart(chart, use_container_width=True)

else:
    st.warning("Please select at least one feature and a target variable.")
