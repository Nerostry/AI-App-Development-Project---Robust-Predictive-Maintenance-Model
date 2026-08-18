import math
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(title="Predictive Maintenance API")

# Define request body structure (optional schema validation)
class PredictionInput(BaseModel):
    machineID: int = 1
    model: str = "model3"
    age: int = 18

def sanitize_data(data):
    """Recursively convert float NaN/Inf to 0.0 for JSON compliance."""
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return 0.0
    return data

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}

@app.post("/predict")
def predict(data: dict):
    # Sanitize inputs to prevent float NaN/Inf JSON errors
    cleaned_input = sanitize_data(data)
    
    # Place model inference logic here using cleaned_input
    sample_prediction = {
        "status": "Success",
        "predicted_RUL": 142,
        "maintenance_required": False
    }
    return sample_prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)