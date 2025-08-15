import streamlit as st
import pandas as pd
import numpy as np

# --------------------------
# Helper Functions
# --------------------------
def train_test_split_manual(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def linear_regression_manual(X, y):
    # Adding intercept
    X_b = np.c_[np.ones((len(X), 1)), X]
    # Normal Equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

def predict_manual(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b.dot(theta)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Power Demand Forecast", layout="wide")

# Sidebar
st.sidebar.header("⚙️ Model Configuration")
weight_temp = st.sidebar.slider("Weight: Temperature", 0.0, 2.0, 1.0, 0.1)
weight_industry = st.sidebar.slider("Weight: Industry Output", 0.0, 2.0, 1.0, 0.1)
weight_population = st.sidebar.slider("Weight: Population", 0.0, 2.0, 1.0, 0.1)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, 5) / 100

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    # Feature selection
    features = ["Temperature", "IndustryOutput", "Population"]
    if not all(f in df.columns for f in features + ["PowerDemand"]):
        st.error(f"CSV must contain columns: {features + ['PowerDemand']}")
    else:
        # Apply weights
        df["Temperature"] *= weight_temp
        df["IndustryOutput"] *= weight_industry
        df["Population"] *= weight_population

        X = df[features].values
        y = df["PowerDemand"].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=test_size)

        # Train model
        theta = linear_regression_manual(X_train, y_train)

        # Predictions
        y_pred_train = predict_manual(X_train, theta)
        y_pred_test = predict_manual(X_test, theta)

        # Metrics
        train_mape = mape(y_train, y_pred_train)
        test_mape = mape(y_test, y_pred_test)
        test_rmse = rmse(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        # Output
        st.subheader("📊 Model Coefficients")
        st.write({"Intercept": theta[0], "Temp": theta[1], "Industry": theta[2], "Population": theta[3]})

        st.subheader("📈 Model Performance")
        st.write(f"Train MAPE: {train_mape:.2f}%")
        st.write(f"Test MAPE: {test_mape:.2f}%")
        st.write(f"Test RMSE: {test_rmse:.2f}")
        st.write(f"Test R²: {test_r2:.2f}")

        # Show predictions
        results_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred_test
        })
        st.subheader("🔍 Predictions vs Actual")
        st.dataframe(results_df)

else:
    st.info("Upload a CSV file to get started. Columns needed: Temperature, IndustryOutput, Population, PowerDemand")
