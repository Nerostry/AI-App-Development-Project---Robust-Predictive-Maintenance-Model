# pipelines/model_save.py
"""
Standalone utility to save a trained PyTorch model's state_dict
to the saved_models/ directory. Trains a fresh model via
model_train.train_model() and immediately persists it, OR can be
imported and called with an already-trained model instance.
"""

import os
from pathlib import Path
import torch

from pipelines.model_train import StreamingMultimodalTransformer, train_model

# ==========================================
# Config: match paths used elsewhere in the pipeline
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "saved_models"
MODEL_FILENAME = "predictive_maintenance_model.pth"
OUTPUT_PATH = OUTPUT_DIR / MODEL_FILENAME


def save_model(model: torch.nn.Module, output_dir: Path = OUTPUT_DIR, filename: str = MODEL_FILENAME) -> Path:
    """Save a PyTorch model's state_dict to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / filename

    torch.save(model.state_dict(), model_path)
    print(f"PyTorch model state saved successfully to {model_path}")
    return model_path


if __name__ == "__main__":
    # Option A (default): train a fresh model right here, then save it.
    trained_model, feature_cols = train_model(save_path=OUTPUT_PATH)
    save_model(trained_model)

    # Option B: if you already have a trained model object in memory
    # (e.g. passed in from run_pipeline.py), just call:
    #     save_model(my_trained_model)
    # instead of retraining from scratch.