import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/solar_model.pkl")

st.title("☀️ Solar Supply Predictor")
st.write("Predict solar power generation (kWh) based on weather conditions")

# Input fields
temp = st.slider("Temperature (°C)", 10, 50, 30)
sunlight = st.slider("Sunlight Hours", 0, 15, 8)
cloud = st.slider("Cloud Cover (%)", 0, 100, 20)

if st.button("Predict"):
    data = pd.DataFrame({
        'Temperature': [temp],
        'SunlightHours': [sunlight],
        'CloudCover': [cloud]
    })
    result = model.predict(data)[0]
    st.success(f"Predicted Solar Output: {result:.2f} kWh")
