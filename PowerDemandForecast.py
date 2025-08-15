import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helper: Simple Linear Regression
# -------------------------------
class SimpleLinearRegression:
    def fit(self, X, y):
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        self.coef_ = np.cov(X, y, bias=True)[0, 1] / np.var(X)
        self.intercept_ = np.mean(y) - self.coef_ * np.mean(X)

    def predict(self, X):
        X = np.array(X).flatten()
        return self.coef_ * X + self.intercept_


# -------------------------------
# Streamlit App
# -------------------------------
st.title("📈 Power Demand Forecasting App")

# Sidebar
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
test_ratio = st.sidebar.slider("Test data ratio", 0.1, 0.9, 0.2, 0.05)

feature_col = st.sidebar.text_input("Feature Column (X)", "Feature")
target_col = st.sidebar.text_input("Target Column (y)", "Target")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    if feature_col in df.columns and target_col in df.columns:
        # Split data
        train_size = int(len(df) * (1 - test_ratio))
        X_train, X_test = df[feature_col][:train_size], df[feature_col][train_size:]
        y_train, y_test = df[target_col][:train_size], df[target_col][train_size:]

        # Train model
        model = SimpleLinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Accuracy metrics
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        st.subheader("📊 Model Performance Metrics")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Graph - Actual vs Predicted
        st.subheader("📉 Actual vs Predicted (Test Data)")
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label="Actual", marker="o")
        ax.plot(y_pred, label="Predicted", marker="x")
        ax.set_xlabel("Test Data Index")
        ax.set_ylabel("Target Value")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Please enter valid column names from your dataset.")
else:
    st.info("Upload a CSV file to begin.")
