"""
predictive_maintenance_service/src/app.py

Fixed inference service that:
1. Uses the model class that actually exists in the repo (TabularSequenceTransformer
   from pipelines/model_train.py) instead of the non-existent StreamingMultimodalTransformer.
2. Loads feature_columns.pkl (saved during training) to know the exact input shape,
   instead of hardcoding TS_FEATURES/NUM_FEATURES/IMG_SIZE.
3. Loads the tuned decision threshold from decision_threshold.pkl (produced by
   pipelines/model_eval.py) instead of hardcoding 0.5.
4. Exposes /health and /ready endpoints so the ingestion -> processing -> AI ->
   UI chain (and Kubernetes readiness probes) can tell whether the model is
   actually usable before sending traffic to this service.
5. Accepts a flat feature vector (matching SEQUENCE_LENGTH x num_features from
   training) so the UI/processor can call it without needing to know internal
   tensor shapes ahead of time -- the service builds the tensor itself.
"""

import os
import math
import logging
from importlib import import_module
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch

# FastAPI is an optional runtime dependency for this inference service. Resolve
# it lazily so importing this module does not produce an unresolved-import
# diagnostic in environments that do not install the service dependencies.
_fastapi = import_module("fastapi")
FastAPI = _fastapi.FastAPI
HTTPException = _fastapi.HTTPException
from pydantic import BaseModel, Field

import uvicorn

# Import the model + constants that are ACTUALLY used in training/eval,
# so this service can never drift out of sync with what was trained.
from pipelines.model_train import (
    TabularSequenceTransformer,
    SEQUENCE_LENGTH,
)

# ==========================================
# Configuration
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictive_maintenance_service")

BASE_DIR = Path(__file__).resolve().parent.parent  # predictive_maintenance_service/
SAVE_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR.parent / "saved_models"))

MODEL_PATH = SAVE_DIR / "predictive_maintenance_model.pth"
FEATURE_COLS_PATH = SAVE_DIR / "feature_columns.pkl"
THRESHOLD_PATH = SAVE_DIR / "decision_threshold.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Global state -- populated on startup, checked by /health and /predict
# ==========================================
model: Optional[torch.nn.Module] = None
feature_cols: Optional[List[str]] = None
decision_threshold: float = 0.5  # fallback if tuned threshold isn't found yet
model_load_error: Optional[str] = None


def model_artifacts_present() -> dict:
    """Reports exactly which required files are missing, so a human debugging
    a broken deployment doesn't have to guess."""
    return {
        "model_weights": MODEL_PATH.exists(),
        "feature_columns": FEATURE_COLS_PATH.exists(),
        "decision_threshold": THRESHOLD_PATH.exists(),  # optional, has a fallback
    }


def load_model_artifacts() -> None:
    """Attempts to load weights + feature columns. Never raises -- failures
    are recorded in model_load_error so /health can report them instead of
    crashing the whole service on startup (which would take down the
    ingestion pipeline's downstream consumer with it)."""
    global model, feature_cols, decision_threshold, model_load_error

    artifacts = model_artifacts_present()

    if not artifacts["model_weights"]:
        model_load_error = f"No model checkpoint found at '{MODEL_PATH}'. Run pipelines/model_train.py first."
        logger.warning(model_load_error)
        return

    if not artifacts["feature_columns"]:
        model_load_error = (
            f"Model weights found but '{FEATURE_COLS_PATH}' is missing. "
            "Cannot determine input feature shape safely. Re-run pipelines/model_train.py "
            "(it saves feature_columns.pkl alongside the weights)."
        )
        logger.warning(model_load_error)
        return

    try:
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        logger.info(f"Loaded {len(feature_cols)} feature columns from '{FEATURE_COLS_PATH}'")

        loaded_model = TabularSequenceTransformer(num_features=len(feature_cols)).to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()

        model = loaded_model
        model_load_error = None
        logger.info(f"Model loaded successfully on device: {device}")

    except Exception as e:
        model = None
        model_load_error = f"Failed to load model: {e}"
        logger.error(model_load_error)
        return

    if THRESHOLD_PATH.exists():
        try:
            decision_threshold = float(joblib.load(THRESHOLD_PATH))
            logger.info(f"Loaded tuned decision threshold: {decision_threshold:.4f}")
        except Exception as e:
            logger.warning(f"Could not load tuned threshold, falling back to 0.5: {e}")
            decision_threshold = 0.5
    else:
        logger.warning(
            f"No tuned threshold at '{THRESHOLD_PATH}' (run pipelines/model_eval.py). "
            f"Using default 0.5 -- this typically gives near-zero recall on imbalanced data."
        )


# ==========================================
# FastAPI App
# ==========================================
app = FastAPI(title="Predictive Maintenance Inference Service")


@app.on_event("startup")
def on_startup():
    load_model_artifacts()


# ==========================================
# Request / Response Schemas
# ==========================================
class PredictionInput(BaseModel):
    # One sequence window of shape [SEQUENCE_LENGTH, num_features], where
    # num_features must match len(feature_cols) from training.
    sequence: List[List[float]] = Field(
        ...,
        description=f"Shape [{SEQUENCE_LENGTH}, num_features] window of engineered features, "
                    f"in the SAME column order as pipelines/model_train.py's feature_columns.pkl",
    )


class PredictionOutput(BaseModel):
    status: str
    failure_probability: float
    maintenance_required: bool
    decision_threshold_used: float


def sanitize_value(value: float) -> float:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return value


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
    """Liveness probe -- process is up, regardless of model state."""
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """Readiness probe -- use this in Kubernetes/docker-compose healthchecks
    so traffic isn't routed here until the model is actually usable.
    Reports exactly what's missing instead of a generic failure."""
    artifacts = model_artifacts_present()
    is_ready = model is not None

    response = {
        "ready": is_ready,
        "model_loaded": model is not None,
        "artifacts_found": artifacts,
        "decision_threshold": decision_threshold,
        "num_features_expected": len(feature_cols) if feature_cols else None,
        "sequence_length_expected": SEQUENCE_LENGTH,
    }
    if not is_ready:
        response["error"] = model_load_error
        raise HTTPException(status_code=503, detail=response)
    return response


@app.post("/reload")
def reload_model():
    """Manually re-trigger artifact loading, e.g. after training finishes
    and drops fresh weights into saved_models/ without restarting the pod."""
    load_model_artifacts()
    return {"model_loaded": model is not None, "error": model_load_error}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if model is None or feature_cols is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model is not ready to serve predictions.",
                "reason": model_load_error,
                "artifacts_found": model_artifacts_present(),
            },
        )

    num_features = len(feature_cols)

    if len(data.sequence) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"'sequence' must have exactly {SEQUENCE_LENGTH} time steps, got {len(data.sequence)}",
        )
    if any(len(row) != num_features for row in data.sequence):
        raise HTTPException(
            status_code=400,
            detail=f"Each row in 'sequence' must have exactly {num_features} features "
                   f"(matching training feature_columns.pkl), got mismatched row length(s).",
        )

    try:
        tensor = torch.tensor([data.sequence], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probability = torch.sigmoid(logits).item()

        probability = sanitize_value(probability)

        return PredictionOutput(
            status="Success",
            failure_probability=round(probability, 4),
            maintenance_required=probability >= decision_threshold,
            decision_threshold_used=round(decision_threshold, 4),
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)