import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

from pipelines.model_train import MultiModalPredictiveMaintenanceModel, MaintenanceMultimodalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "model_train_dataset"
TEST_DATASET_PATH = DATASET_DIR / "test_scaled.csv"
SAVE_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVE_DIR / "predictive_maintenance_model.pth"
FEATURE_COLS_PATH = SAVE_DIR / "feature_columns.pkl"
TRAINING_CONFIG_PATH = SAVE_DIR / "training_config.pkl"
THRESHOLD_PATH = SAVE_DIR / "decision_threshold.pkl"


def load_test_dataset(training_config):
    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(f"Could not find test dataset at {TEST_DATASET_PATH.resolve()}")
    df = pd.read_csv(TEST_DATASET_PATH)
    print(f"Loaded test dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    feature_cols = training_config["feature_columns"]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Test schema is incompatible with training. Missing feature columns: "
            + ", ".join(missing_cols)
        )

    unexpected = [c for c in df.columns if c in {"target_failure_window", "failed", "label", "target", "class", "is_failed"}]
    if not unexpected:
        raise ValueError("Test dataset does not contain a recognized target column.")

    from pipelines.model_train import resolve_label_column
    labels, _ = resolve_label_column(df)

    machine_id_col = "machineID" if "machineID" in df.columns else ("machine_id" if "machine_id" in df.columns else None)
    dataset = MaintenanceMultimodalDataset(
        df=df,
        telemetry_cols=training_config["telemetry_columns"],
        event_cols=training_config["event_columns"],
        static_cols=training_config["static_columns"],
        labels=labels,
        sequence_length=training_config["sequence_length"],
        step=training_config["window_step"],
        machine_id_col=machine_id_col,
    )
    return dataset


def evaluate_model(model, dataset, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda", num_workers=0)
    all_targets, all_probs = [], []

    with torch.inference_mode():
        for telemetry, events, static, labels in loader:
            telemetry = telemetry.to(device, non_blocking=True)
            events = events.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            probabilities = torch.sigmoid(model(telemetry, events, static))
            all_probs.append(probabilities.cpu().numpy())
            all_targets.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets).astype(int)
    pr_auc = average_precision_score(all_targets, all_probs)

    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_values))
    optimal_threshold = float(thresholds[best_idx])
    predictions = (all_probs >= optimal_threshold).astype(int)

    tuned_precision = precision_score(all_targets, predictions, zero_division=0)
    tuned_recall = recall_score(all_targets, predictions, zero_division=0)
    tuned_f1 = f1_score(all_targets, predictions, zero_division=0)

    print("\n" + "=" * 55)
    print("       MULTIMODAL MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"Device                       : {device}")
    print(f"Test sequences               : {len(dataset):,}")
    print(f"Positive rate                : {all_targets.mean():.4%}")
    print(f"PR-AUC                       : {pr_auc:.4f}")
    print(f"Optimal threshold            : {optimal_threshold:.4f}")
    print(f"F1                           : {tuned_f1:.4f}")
    print(f"Precision                    : {tuned_precision:.4f}")
    print(f"Recall                       : {tuned_recall:.4f}")
    print("=" * 55)

    return {
        "pr_auc": float(pr_auc),
        "threshold": optimal_threshold,
        "f1": float(tuned_f1),
        "precision": float(tuned_precision),
        "recall": float(tuned_recall),
    }


def save_threshold(threshold):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(threshold, THRESHOLD_PATH)
    print(f"\nSaved decision threshold to:\n{THRESHOLD_PATH}")


if __name__ == "__main__":
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"Missing: {FEATURE_COLS_PATH}")
    if not TRAINING_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing: {TRAINING_CONFIG_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")

    feature_cols = joblib.load(FEATURE_COLS_PATH)
    training_config = joblib.load(TRAINING_CONFIG_PATH)
    print(f"Loaded {len(feature_cols)} feature columns.")
    print(f"Model type: {training_config.get('model_type', 'unknown')}")
    print(f"Telemetry: {training_config['telemetry_dim']} | Events: {training_config['event_dim']} | Static: {training_config['static_dim']}")

    model = MultiModalPredictiveMaintenanceModel(
        telemetry_dim=training_config["telemetry_dim"],
        event_dim=training_config["event_dim"],
        static_dim=training_config["static_dim"],
        embed_dim=training_config.get("model_embed_dim", 96),
        nhead=training_config.get("model_nhead", 4),
        num_layers=training_config.get("model_num_layers", 3),
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded model from:\n{MODEL_PATH}")

    test_dataset = load_test_dataset(training_config)
    results = evaluate_model(model, test_dataset)
    save_threshold(results["threshold"])
