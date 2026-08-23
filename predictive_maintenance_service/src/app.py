import os
import math
import logging
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipelines.model_train import StreamingMultimodalTransformer

# ==========================================
# Configuration
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictive_maintenance_service")

MODEL_DIR = os.getenv("MODEL_DIR", "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "predictive_maintenance_model.pth")

SEQUENCE_LENGTH = 64
TS_FEATURES = 4
NUM_FEATURES = 5
IMG_SIZE = (3, 64, 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# FastAPI App
# ==========================================
app = FastAPI(title="Predictive Maintenance Inference Service")

model: StreamingMultimodalTransformer | None = None


@app.on_event("startup")
def load_model():
    """Loads the trained model weights from the saved_models directory on startup."""
    global model

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"No model checkpoint found at '{MODEL_PATH}'. "
            "The /predict endpoint will return errors until a model is available."
        )
        return

    logger.info(f"Loading model checkpoint from '{MODEL_PATH}'...")
    loaded_model = StreamingMultimodalTransformer(
        ts_features=TS_FEATURES,
        num_features=NUM_FEATURES,
    ).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    loaded_model.eval()

    model = loaded_model
    logger.info(f"Model loaded successfully on device: {device}")


# ==========================================
# Request / Response Schemas
# ==========================================
class PredictionInput(BaseModel):
    timeseries: List[List[float]] = Field(
        ..., description=f"Shape [{SEQUENCE_LENGTH}, {TS_FEATURES}] sensor time-series window"
    )
    numerical: List[float] = Field(
        ..., description=f"Length {NUM_FEATURES} numerical feature vector"
    )
    image: List[List[List[float]]] = Field(
        ..., description=f"Shape {list(IMG_SIZE)} image tensor (C, H, W)"
    )


class PredictionOutput(BaseModel):
    status: str
    failure_probability: float
    predicted_ttf_hours: float
    maintenance_required: bool


# ==========================================
# Helpers
# ==========================================
def sanitize_value(value: float) -> float:
    """Converts NaN/Inf to 0.0 for JSON compliance."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return value


def validate_shapes(data: PredictionInput):
    if len(data.timeseries) != SEQUENCE_LENGTH or any(
        len(row) != TS_FEATURES for row in data.timeseries
    ):
        raise HTTPException(
            status_code=400,
            detail=f"timeseries must have shape [{SEQUENCE_LENGTH}, {TS_FEATURES}]",
        )
    if len(data.numerical) != NUM_FEATURES:
        raise HTTPException(
            status_code=400, detail=f"numerical must have length {NUM_FEATURES}"
        )
    expected_c, expected_h, expected_w = IMG_SIZE
    if (
        len(data.image) != expected_c
        or any(len(row) != expected_h for row in data.image)
        or any(len(px) != expected_w for row in data.image for px in row)
    ):
        raise HTTPException(
            status_code=400, detail=f"image must have shape {list(IMG_SIZE)}"
        )


# ==========================================
# Routes
# ==========================================
@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance Inference Service is running",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Expected checkpoint at '{MODEL_PATH}'.",
        )

    validate_shapes(data)

    try:
        ts_tensor = torch.tensor([data.timeseries], dtype=torch.float32).to(device)
        num_tensor = torch.tensor([data.numerical], dtype=torch.float32).to(device)
        img_tensor = torch.tensor([data.image], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits, ttf_pred = model(ts_tensor, num_tensor, img_tensor)
            probability = torch.sigmoid(logits).item()
            ttf_hours = ttf_pred.item()

        probability = sanitize_value(probability)
        ttf_hours = sanitize_value(ttf_hours)

        return PredictionOutput(
            status="Success",
            failure_probability=round(probability, 4),
            predicted_ttf_hours=round(ttf_hours, 2),
            maintenance_required=probability >= 0.5,
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)