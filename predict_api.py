from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")

class TrafficData(BaseModel):
    Active_users: int
    New_users: int
    Message_rate: float
    Media_sharing: float
    Spam_ratio: float
    User_sentiment: int
    Server_load: int
    Time_of_day: int
    Latency: int
    Bandwidth_usage: int

@app.get("/")
def root():
    return {"message": "API is working!"}

@app.post("/predict")
def predict(data: TrafficData):
    input_df = pd.DataFrame([data.dict()])

    # Rename to EXACT column names as used during training
    input_df.rename(columns={
        "Active_users": "Active users",
        "New_users": "New users",
        "Message_rate": "Message rate",
        "Media_sharing": "Media sharing",
        "Spam_ratio": "Spam ratio",
        "User_sentiment": "User sentiment",
        "Server_load": "Server load",
        "Time_of_day": "Time of day",
        "Latency": "Latency",
        "Bandwidth_usage": "Bandwidth usage"
    }, inplace=True)

    print("📥 Renamed input DataFrame:\n", input_df)

    input_scaled = scaler.transform(input_df)
    print("📊 Scaled input data:", input_scaled)

    prediction = model.predict(input_scaled)
    label_map = {0: "Traffic is high 🔴", 1: "Traffic is Low 🟢 ", 2: "Traffic is Medium 🟠"}

    return {"Prediction": label_map.get(int(prediction[0]), "Unknown")}
