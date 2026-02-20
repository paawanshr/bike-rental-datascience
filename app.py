import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    data = pd.read_csv("data/hour.csv")
    return model, scaler, features, data

model, scaler, features, df = load_assets()

df['dteday'] = pd.to_datetime(df['dteday'])


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Introduction",
        "Data Visualization (EDA)",
        "Model & Feature Engineering",
        "Bike Rental Prediction"
    ]
)


if page == "Introduction":
    st.title("🚲 Bike Rental Time Series Project")

    st.write("""
    This project predicts **hourly bike rentals** using historical usage and weather data.
    
    - Dataset: Bike Sharing Dataset
    - Problem Type: Time Series Regression
    - Model: Random Forest Regressor
    - Deployment: Streamlit
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

#eda

elif page == "Data Visualization (EDA)":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Average Rentals by Hour")
    hourly_avg = df.groupby("hr")["cnt"].mean()
    fig, ax = plt.subplots()
    ax.plot(hourly_avg)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Rentals")
    st.pyplot(fig)

    st.subheader("Average Rentals by Day of Week")
    weekday_avg = df.groupby("weekday")["cnt"].mean()
    fig, ax = plt.subplots()
    ax.plot(weekday_avg)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Rentals")
    st.pyplot(fig)

    st.subheader("Seasonal Pattern")
    season_avg = df.groupby("season")["cnt"].mean()
    fig, ax = plt.subplots()
    ax.bar(season_avg.index, season_avg.values)
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Rentals")
    st.pyplot(fig)

#model explanation and feature engineering

elif page == "Model & Feature Engineering":
    st.title("🧠 Model & Feature Engineering")

    st.subheader("Feature Engineering Performed")
    st.markdown("""
    - Lag features (previous hours)
    - Rolling mean and standard deviation
    - Time-based features (hour, weekday, month)
    - Weather variables
    """)

    st.subheader("Why Random Forest?")
    st.write("""
    Random Forest captures non-linear patterns and works well on tabular data.
    It is robust and does not assume linear relationships.
    """)

    st.subheader("Evaluation Metrics Used")
    st.write("""
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R² Score
    """)

# Prediction

elif page == "Bike Rental Prediction":
    st.title("🔮 Predict Bike Rentals")

    st.sidebar.subheader("Input Features")

    inputs = {}
    for feature in features:
        inputs[feature] = st.sidebar.number_input(feature, value=0.0)

    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"🚴 Predicted Bike Rentals: {prediction[0]:.2f}")
