import pandas as pd
import joblib

# Load model
model = joblib.load("models/solar_model.pkl")

def predict_solar_output(temp, sunlight, cloud):
    data = pd.DataFrame({
        'Temperature': [temp],
        'SunlightHours': [sunlight],
        'CloudCover': [cloud]
    })
    prediction = model.predict(data)[0]
    return round(prediction, 2)

# Example
if __name__ == "__main__":
    print("Predicted Solar Output:", predict_solar_output(30, 8, 20), "kWh")
