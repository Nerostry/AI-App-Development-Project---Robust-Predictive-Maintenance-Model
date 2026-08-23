import os
import math
import logging
from typing import List

import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipelines.model_train import TabularSequenceTransformer, SEQUENCE_LENGTH

# ==========================================
# Configuration
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictive_maintenance_service")

MODEL_DIR = os.getenv("MODEL_DIR", "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "predictive_maintenance_model.pth")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "decision_threshold.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# FastAPI App
# ==========================================
app = FastAPI(title="Predictive Maintenance Inference Service")

model: TabularSequenceTransformer | None = None
feature_cols: list | None = None
decision_threshold: float = 0.5


@app.on_event("startup")
def load_model():
    """Loads the trained model weights, feature column order, and tuned
    decision threshold from the saved_models directory on startup."""
    global model, feature_cols, decision_threshold

    if not os.path.exists(FEATURE_COLS_PATH):
        logger.warning(
            f"No feature columns file found at '{FEATURE_COLS_PATH}'. "
            "The /predict endpoint will return errors until training artifacts are available."
        )
        return

    feature_cols = joblib.load(FEATURE_COLS_PATH)
    logger.info(f"Loaded {len(feature_cols)} feature columns.")

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"No model checkpoint found at '{MODEL_PATH}'. "
            "The /predict endpoint will return errors until a model is available."
        )
        return

    logger.info(f"Loading model checkpoint from '{MODEL_PATH}'...")
    loaded_model = TabularSequenceTransformer(num_features=len(feature_cols)).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    loaded_model.eval()

    model = loaded_model
    logger.info(f"Model loaded successfully on device: {device}")

    if os.path.exists(THRESHOLD_PATH):
        decision_threshold = joblib.load(THRESHOLD_PATH)
        logger.info(f"Loaded tuned decision threshold: {decision_threshold:.4f}")
    else:
        logger.info(f"No tuned threshold found; defaulting to {decision_threshold}")


# ==========================================
# Request / Response Schemas
# ==========================================
class PredictionInput(BaseModel):
    features: List[List[float]] = Field(
        ..., description=f"Shape [{SEQUENCE_LENGTH}, num_features] sensor sequence window, "
                          "columns must match the training feature order"
    )


class PredictionOutput(BaseModel):
    status: str
    failure_probability: float
    maintenance_required: bool
    decision_threshold: float


# ==========================================
# Helpers
# ==========================================
def sanitize_value(value: float) -> float:
    """Converts NaN/Inf to 0.0 for JSON compliance."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return value


def validate_shape(data: PredictionInput):
    if feature_cols is None:
        raise HTTPException(status_code=503, detail="Feature columns not loaded.")
    if len(data.features) != SEQUENCE_LENGTH or any(
        len(row) != len(feature_cols) for row in data.features
    ):
        raise HTTPException(
            status_code=400,
            detail=f"features must have shape [{SEQUENCE_LENGTH}, {len(feature_cols)}]",
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

    validate_shape(data)

    try:
        features_tensor = torch.tensor([data.features], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(features_tensor)
            probability = torch.sigmoid(logits).item()

        probability = sanitize_value(probability)

        return PredictionOutput(
            status="Success",
            failure_probability=round(probability, 4),
            maintenance_required=probability >= decision_threshold,
            decision_threshold=round(decision_threshold, 4),
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)