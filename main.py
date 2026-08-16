from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(title="Predictive Maintenance API")

# Define request body structure to match incoming data
class PredictionInput(BaseModel):
    machineID: int = 1
    model: str = "model3"
    age: int = 18

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}

@app.post("/predict")
def predict(data: dict):
    # Process incoming sensor/engine data from Streamlit UI
    # Replace this mock logic with your actual model inference
    sample_prediction = {
        "status": "Success",
        "predicted_RUL": 142,
        "maintenance_required": False
    }
    return sample_prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)