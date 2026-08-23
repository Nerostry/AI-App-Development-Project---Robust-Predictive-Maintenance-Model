import sys
import os

# Ensure project root is on sys.path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from pipelines.model_train import (
    TabularSequenceTransformer,
    MaintenanceSequenceDataset,
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "model_train_dataset"
)

TEST_DATASET_PATH = (
    DATASET_DIR
    / "test_scaled.csv"
)

SAVE_DIR = (
    BASE_DIR
    / "saved_models"
)

MODEL_PATH = (
    SAVE_DIR
    / "predictive_maintenance_model.pth"
)

FEATURE_COLS_PATH = (
    SAVE_DIR
    / "feature_columns.pkl"
)

TRAINING_CONFIG_PATH = (
    SAVE_DIR
    / "training_config.pkl"
)

THRESHOLD_PATH = (
    SAVE_DIR
    / "decision_threshold.pkl"
)


# ============================================================
# Load Test Dataset
# ============================================================

def load_test_dataset(
    feature_cols,
    sequence_length,
    window_step,
):
    """
    Loads the test dataset using exactly the same
    feature order and sequence configuration as training.
    """

    if not TEST_DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Could not find test dataset at "
            f"{TEST_DATASET_PATH.resolve()}"
        )

    df = pd.read_csv(
        TEST_DATASET_PATH
    )

    print(
        f"Loaded test dataset: "
        f"{df.shape[0]:,} rows, "
        f"{df.shape[1]} columns."
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    from pipelines.model_train import (
        resolve_label_column
    )

    labels = resolve_label_column(
        df
    )

    # --------------------------------------------------------
    # Ensure feature compatibility
    # --------------------------------------------------------

    missing_cols = [
        c
        for c in feature_cols
        if c not in df.columns
    ]

    if missing_cols:

        print(
            f"[WARN] Test set is missing "
            f"{len(missing_cols)} feature columns."
        )

        print(
            f"Filling missing columns with 0: "
            f"{missing_cols}"
        )

        for col in missing_cols:
            df[col] = 0

    # --------------------------------------------------------
    # Machine ID
    # --------------------------------------------------------

    if "machineID" in df.columns:

        machine_id_col = "machineID"

    elif "machine_id" in df.columns:

        machine_id_col = "machine_id"

    else:

        machine_id_col = None

        print(
            "[WARN] No machine ID column found."
        )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    dataset = MaintenanceSequenceDataset(
        df=df,
        feature_cols=feature_cols,
        labels=labels,
        sequence_length=sequence_length,
        step=window_step,
        machine_id_col=machine_id_col,
    )

    return dataset


# ============================================================
# Evaluate
# ============================================================

def evaluate_model(
    model,
    dataset,
    batch_size=256,
):
    """
    Runs model inference and calculates
    threshold-independent and threshold-dependent metrics.
    """

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        ),
        num_workers=0,
    )

    all_targets = []
    all_probs = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for features, labels in loader:

            features = features.to(
                device,
                non_blocking=True
            )

            logits = model(
                features
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_probs.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                labels.numpy()
            )

    all_probs = np.concatenate(
        all_probs
    )

    all_targets = np.concatenate(
        all_targets
    ).astype(int)

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        all_targets,
        all_probs
    )

    # --------------------------------------------------------
    # Find F1-optimal threshold
    # --------------------------------------------------------

    precision, recall, thresholds = (
        precision_recall_curve(
            all_targets,
            all_probs
        )
    )

    # Precision/recall contain one more
    # element than thresholds.

    f1_values = (
        2
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_idx = int(
        np.argmax(f1_values)
    )

    optimal_threshold = float(
        thresholds[best_idx]
    )

    best_f1 = float(
        f1_values[best_idx]
    )

    # --------------------------------------------------------
    # Predictions using tuned threshold
    # --------------------------------------------------------

    predictions = (
        all_probs
        >= optimal_threshold
    ).astype(int)

    tuned_precision = precision_score(
        all_targets,
        predictions,
        zero_division=0,
    )

    tuned_recall = recall_score(
        all_targets,
        predictions,
        zero_division=0,
    )

    tuned_f1 = f1_score(
        all_targets,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 55
    )

    print(
        "          MODEL EVALUATION RESULTS"
    )

    print(
        "=" * 55
    )

    print(
        f"Device                       : "
        f"{device}"
    )

    print(
        f"Test sequences               : "
        f"{len(dataset):,}"
    )

    print(
        f"Positive rate                : "
        f"{all_targets.mean():.4%}"
    )

    print(
        f"PR-AUC                       : "
        f"{pr_auc:.4f}"
    )

    print(
        f"Optimal threshold            : "
        f"{optimal_threshold:.4f}"
    )

    print(
        f"F1                           : "
        f"{tuned_f1:.4f}"
    )

    print(
        f"Precision                    : "
        f"{tuned_precision:.4f}"
    )

    print(
        f"Recall                       : "
        f"{tuned_recall:.4f}"
    )

    print(
        "=" * 55
    )

    return {
        "pr_auc": float(pr_auc),
        "threshold": optimal_threshold,
        "f1": tuned_f1,
        "precision": tuned_precision,
        "recall": tuned_recall,
    }


# ============================================================
# Save Threshold
# ============================================================

def save_threshold(
    threshold,
):
    """
    Saves the decision threshold used by
    the inference service.
    """

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        threshold,
        THRESHOLD_PATH
    )

    print(
        f"\nSaved decision threshold to:"
        f"\n{THRESHOLD_PATH}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Load feature columns
    # --------------------------------------------------------

    if not FEATURE_COLS_PATH.exists():

        raise FileNotFoundError(
            f"Missing: "
            f"{FEATURE_COLS_PATH}"
        )

    feature_cols = joblib.load(
        FEATURE_COLS_PATH
    )

    print(
        f"Loaded {len(feature_cols)} "
        f"feature columns."
    )

    # --------------------------------------------------------
    # 2. Load training configuration
    # --------------------------------------------------------

    if not TRAINING_CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Missing: "
            f"{TRAINING_CONFIG_PATH}"
        )

    training_config = joblib.load(
        TRAINING_CONFIG_PATH
    )

    sequence_length = training_config[
        "sequence_length"
    ]

    window_step = training_config[
        "window_step"
    ]

    print(
        f"Sequence length: "
        f"{sequence_length}"
    )

    print(
        f"Window step: "
        f"{window_step}"
    )

    # --------------------------------------------------------
    # 3. Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Missing trained model: "
            f"{MODEL_PATH}"
        )

    model = (
        TabularSequenceTransformer(
            num_features=len(
                feature_cols
            )
        )
        .to(device)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    print(
        f"Loaded model from:"
        f"\n{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # 4. Build test dataset
    # --------------------------------------------------------

    test_dataset = load_test_dataset(
        feature_cols=feature_cols,
        sequence_length=sequence_length,
        window_step=window_step,
    )

    # --------------------------------------------------------
    # 5. Evaluate
    # --------------------------------------------------------

    results = evaluate_model(
        model=model,
        dataset=test_dataset,
        batch_size=256,
    )

    # --------------------------------------------------------
    # 6. Save threshold
    # --------------------------------------------------------

    save_threshold(
        results["threshold"]
    )