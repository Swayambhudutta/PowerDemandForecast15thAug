import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Power Demand Forecasting", layout="wide")

# Sidebar
st.sidebar.title("Model Settings")
weight_feature_1 = st.sidebar.slider("Weightage for Feature 1", 0.0, 2.0, 1.0, 0.1)
weight_feature_2 = st.sidebar.slider("Weightage for Feature 2", 0.0, 2.0, 1.0, 0.1)
forecast_periods = st.sidebar.number_input("Forecast Periods (Days)", min_value=1, max_value=365, value=30)
test_size_ratio = st.sidebar.slider("Test Data Ratio", 0.1, 0.5, 0.2, 0.05)

# Upload file
st.title("📊 Power Demand Forecasting Tool")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Ensure there are at least 3 columns
    if df.shape[1] < 3:
        st.error("Please upload a dataset with at least 3 columns: [Target, Feature1, Feature2].")
    else:
        target_col = df.columns[0]
        feature_cols = df.columns[1:3]

        # Apply weightages
        df["Weighted_Feature1"] = df[feature_cols[0]] * weight_feature_1
        df["Weighted_Feature2"] = df[feature_cols[1]] * weight_feature_2

        # Prepare data
        X = df[["Weighted_Feature1", "Weighted_Feature2"]]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, shuffle=False)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mae = np.mean(np.abs(y_test - predictions))

        st.subheader("📈 Model Performance Metrics")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Actual vs Predicted plot
        st.subheader("📊 Actual vs. Predicted on Test Data")
        fig1, ax1 = plt.subplots()
        ax1.plot(y_test.values, label="Actual", marker='o')
        ax1.plot(predictions, label="Predicted", marker='x')
        ax1.set_title("Actual vs Predicted")
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Power Demand")
        ax1.legend()
        st.pyplot(fig1)

        # Forecast future
        st.subheader("🔮 Future Forecast")
        last_features = X.iloc[-1].values.reshape(1, -1)
        forecast_values = []
        for _ in range(forecast_periods):
            next_pred = model.predict(last_features)[0]
            forecast_values.append(next_pred)
            last_features = np.array([[last_features[0][0], last_features[0][1]]])  # keeping same features for demo

        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_periods)
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast_values})

        fig2, ax2 = plt.subplots()
        ax2.plot(forecast_df["Date"], forecast_df["Forecast"], marker='o', color='orange')
        ax2.set_title("Future Power Demand Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Forecasted Demand")
        st.pyplot(fig2)

        st.subheader("📅 Forecast Data")
        st.dataframe(forecast_df)
else:
    st.info("Please upload a CSV file to proceed.")
