import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Ensure model folder exists
os.makedirs("models", exist_ok=True)

# Load your dataset
data = pd.read_csv("data/solar_data.csv")

# Replace with your real column names
X = data[["temperature", "humidity", "irradiance"]]
y = data["solar_output"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save model to models/ folder
joblib.dump(model, "models/solar_model.pkl")

print("✅ Model trained and saved at models/solar_model.pkl")
