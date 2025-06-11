from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")

# FastAPI app
app = FastAPI()

# Define input schema
class TrafficData(BaseModel):
    Active_Users: int
    New_Users: int
    Message_Rate: float
    Media_Sharing: int
    Spam_Ratio: float
    Sentiment: int
    Server_Load: int
    Time_of_Day: int
    Latency: int
    Bandwidth_Usage: int

@app.post("/predict")
def predict(data: TrafficData):
    input_data = np.array([[ 
        data.Active_Users, data.New_Users, data.Message_Rate,
        data.Media_Sharing, data.Spam_Ratio, data.Sentiment,
        data.Server_Load, data.Time_of_Day, data.Latency, data.Bandwidth_Usage
    ]])
    
    print("📥 Raw input data:", input_data)
    input_scaled = scaler.transform(input_data)
    print("📊 Scaled input data:", input_scaled)

    prediction = model.predict(input_scaled)
    print("🔮 Raw model prediction:", prediction)

    # Mapping numeric prediction to readable labels with emojis
    label_map = {
        0: "Traffic is Low 🟢",
        1: "Traffic is Medium 🟠",
        2: "Traffic is High 🔴"
    }

    return {"Prediction": label_map.get(int(prediction[0]), "Unknown")}
