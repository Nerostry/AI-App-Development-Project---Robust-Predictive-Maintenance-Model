import sys
import os

# Ensure the project root (parent of this file's directory) is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score

from pipelines.model_train import (
    TabularSequenceTransformer,
    MaintenanceSequenceDataset,
    resolve_label_column,
    SEQUENCE_LENGTH,
    WINDOW_STEP,
    BATCH_SIZE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Paths (must match model_train.py / model_save.py)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "model_train_dataset"
TEST_DATASET_PATH = DATASET_DIR / "test_scaled.csv"

SAVE_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVE_DIR / "predictive_maintenance_model.pth"
FEATURE_COLS_PATH = SAVE_DIR / "feature_columns.pkl"


def load_test_dataset(feature_cols: list) -> MaintenanceSequenceDataset:
    """Loads test_scaled.csv and builds sequence windows using the SAME
    feature columns (and order) used during training."""
    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Could not find test dataset at {TEST_DATASET_PATH.resolve()}. "
            "Did you run feature_engineer.py to produce train/test splits?"
        )

    df = pd.read_csv(TEST_DATASET_PATH)
    print(f"Loaded test dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    labels = resolve_label_column(df)

    machine_id_col = "machineID" if "machineID" in df.columns else "machine_id"
    has_machine_id = machine_id_col in df.columns

    # Make sure every expected feature column exists in the test set;
    # fill any missing one-hot dummy columns (e.g. a model_X category
    # absent from the test split) with zeros so shapes line up.
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[WARN] Test set missing {len(missing_cols)} training feature columns; filling with 0: {missing_cols}")
        for c in missing_cols:
            df[c] = 0

    dataset = MaintenanceSequenceDataset(
        df=df,
        feature_cols=feature_cols,
        labels=labels,
        sequence_length=SEQUENCE_LENGTH,
        step=WINDOW_STEP,
        has_machine_id=has_machine_id,
        machine_id_col=machine_id_col,
    )
    return dataset


def evaluate_and_tune(model: torch.nn.Module, dataset: MaintenanceSequenceDataset):
    """Runs inference over the test set and computes PR-AUC plus an
    F1-optimal decision threshold, as called for in the project report."""
    model.eval()
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_targets, all_probs = [], []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch["label"].numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Precision-Recall AUC (primary metric per report)
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recall, precision)

    # F1-optimal threshold tuning (default 0.5 gives near-zero recall on imbalanced data)
    f1_scores = [2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    predictions = (all_probs >= optimal_threshold).astype(int)
    tuned_precision = precision_score(all_targets, predictions, zero_division=0)
    tuned_recall = recall_score(all_targets, predictions, zero_division=0)

    print("\n" + "=" * 45)
    print("      EVALUATION & THRESHOLD RESULTS       ")
    print("=" * 45)
    print(f"Positive rate (test)         : {all_targets.mean():.4f}")
    print(f"PR-AUC (Average Precision)   : {pr_auc:.4f}")
    print(f"Optimal Probability Threshold: {optimal_threshold:.4f}")
    print(f"F1-Score at Threshold        : {best_f1:.4f}")
    print(f"Precision at Threshold       : {tuned_precision:.4f}")
    print(f"Recall at Threshold          : {tuned_recall:.4f}")
    print("=" * 45)

    return best_f1, optimal_threshold, pr_auc


def save_threshold(optimal_threshold: float, output_dir: Path = SAVE_DIR):
    """Persist the tuned threshold so the inference service (predictive_maintenance_service)
    can apply it instead of the default 0.5 cutoff."""
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = output_dir / "decision_threshold.pkl"
    joblib.dump(optimal_threshold, threshold_path)
    print(f"Saved tuned decision threshold to '{threshold_path}'")
    return threshold_path


if __name__ == "__main__":
    # 1. Load feature column order saved during training
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find '{FEATURE_COLS_PATH}'. Run pipelines/model_train.py first."
        )
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    print(f"Loaded {len(feature_cols)} feature columns used during training.")

    # 2. Load trained model weights
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find trained weights at '{MODEL_PATH}'. Run pipelines/model_train.py first."
        )
    model = TabularSequenceTransformer(num_features=len(feature_cols)).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded trained weights from '{MODEL_PATH}'")

    # 3. Build test dataset with matching feature columns
    test_dataset = load_test_dataset(feature_cols)

    # 4. Evaluate + tune threshold
    best_f1, optimal_threshold, pr_auc = evaluate_and_tune(model, test_dataset)

    # 5. Persist the tuned threshold for downstream inference use
    save_threshold(optimal_threshold)