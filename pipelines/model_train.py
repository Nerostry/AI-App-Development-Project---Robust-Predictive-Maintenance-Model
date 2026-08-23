import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# 0. Reproducibility & Performance
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = device.type == "cuda"

# PyTorch 2.x
USE_COMPILE = (
    device.type == "cuda"
    and hasattr(torch, "compile")
)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "model_train_dataset"
)

TRAIN_DATASET_PATH = (
    DATASET_DIR / "train_scaled.csv"
)

SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_SAVE_PATH = (
    SAVE_DIR
    / "predictive_maintenance_model.pth"
)


SEQUENCE_LENGTH = 32

# 16 = 50% overlap
# 32 = no overlap
WINDOW_STEP = 16

BATCH_SIZE = 256
EPOCHS = 15

LEARNING_RATE = 1e-3

NUM_WORKERS = min(
    4,
    os.cpu_count() or 1
)

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 3


# ============================================================
# 2. Load Dataset
# ============================================================

def load_engineered_dataset(
    dataset_path: Path
):

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path.resolve()}"
        )

    df = pd.read_csv(dataset_path)

    print(
        f"Loaded dataset: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]} columns"
    )

    return df


# ============================================================
# 3. Resolve Label
# ============================================================

def resolve_label_column(df):

    target_candidates = [
        "target_failure_window",
        "failed",
        "label",
        "target",
        "class",
        "is_failed",
    ]

    for col in target_candidates:

        if col in df.columns:

            print(
                f"Using label column: {col}"
            )

            return (
                df[col]
                .astype(np.int8)
            )

    comp_failure_cols = [
        c
        for c in df.columns
        if c.startswith("comp_failure_")
    ]

    if comp_failure_cols:

        print(
            "Reconstructing failure label "
            f"from {len(comp_failure_cols)} columns"
        )

        return (
            df[comp_failure_cols]
            .sum(axis=1)
            .gt(0)
            .astype(np.int8)
        )

    raise ValueError(
        "No usable label column found."
    )


# ============================================================
# 4. Resolve Features
# ============================================================

def resolve_feature_columns(
    df,
    label_col_names
):

    exclude = (
        set(label_col_names)
        | {
            "datetime",
            "machineID",
            "machine_id",
            "comp_failure",
        }
    )

    feature_cols = [
        c
        for c in df.columns
        if (
            c not in exclude
            and pd.api.types.is_numeric_dtype(
                df[c]
            )
        )
    ]

    print(
        f"Using {len(feature_cols)} features"
    )

    return feature_cols


# ============================================================
# 5. Efficient Sequence Dataset
# ============================================================

class MaintenanceSequenceDataset(Dataset):

    """
    Memory-efficient sequence dataset.

    Instead of storing every 32-step sequence,
    only stores the start/end indices.

    This avoids making a huge duplicated array
    when windows overlap.
    """

    def __init__(
        self,
        df,
        feature_cols,
        labels,
        sequence_length=32,
        step=16,
        machine_id_col=None,
    ):

        self.sequence_length = sequence_length
        self.step = step

        # Keep data contiguous for faster slicing.
        self.features = np.ascontiguousarray(
            df[feature_cols]
            .to_numpy(dtype=np.float32)
        )

        self.labels = np.ascontiguousarray(
            labels.to_numpy(dtype=np.float32)
        )

        self.indices = []

        # ----------------------------------------------------
        # Build only window indices
        # ----------------------------------------------------

        if (
            machine_id_col is not None
            and machine_id_col in df.columns
        ):

            machine_groups = (
                df.groupby(
                    machine_id_col,
                    sort=False
                ).groups
            )

            for _, group_idx in machine_groups.items():

                idx = np.asarray(
                    sorted(group_idx),
                    dtype=np.int64
                )

                self._add_windows(
                    idx
                )

        else:

            print(
                "[WARN] Machine ID unavailable. "
                "Using global sequence."
            )

            self._add_windows(
                np.arange(
                    len(df),
                    dtype=np.int64
                )
            )

        if not self.indices:

            raise ValueError(
                "No valid sequences generated."
            )

        self.indices = np.asarray(
            self.indices,
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Pre-compute labels
        # ----------------------------------------------------

        self.seq_labels = (
            self.labels[
                self.indices[:, 1] - 1
            ]
        )

        positive = int(
            self.seq_labels.sum()
        )

        total = len(
            self.seq_labels
        )

        negative = total - positive

        print(
            "\nSequence statistics"
            "\n----------------------------"
        )

        print(
            f"Sequences      : {total:,}"
        )

        print(
            f"Positive       : {positive:,}"
        )

        print(
            f"Negative       : {negative:,}"
        )

        print(
            f"Positive rate  : "
            f"{positive / total:.4%}"
        )

        print(
            f"Sequence length: "
            f"{sequence_length}"
        )

        print(
            f"Window step    : "
            f"{step}"
        )

        print(
            "----------------------------\n"
        )

    def _add_windows(self, indices):

        n = len(indices)

        if n < self.sequence_length:
            return

        # Number of valid windows.
        max_start = (
            n - self.sequence_length + 1
        )

        for start in range(
            0,
            max_start,
            self.step
        ):

            end = (
                start
                + self.sequence_length
            )

            self.indices.append(
                (
                    indices[start],
                    indices[end - 1] + 1
                )
            )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(self, idx):

        start, end = (
            self.indices[idx]
        )

        sequence = self.features[
            start:end
        ]

        label = self.seq_labels[
            idx
        ]

        return (
            torch.from_numpy(sequence),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# ============================================================
# 6. Transformer
# ============================================================

class TabularSequenceTransformer(
    nn.Module
):

    def __init__(
        self,
        num_features,
        embed_dim=64,
        nhead=4,
        num_layers=2,
    ):

        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                num_features,
                embed_dim
            ),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 2,
                batch_first=True,
                dropout=0.1,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(
                embed_dim,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                1
            ),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return (
            self.classifier_head(x)
            .squeeze(-1)
        )


# ============================================================
# 7. Class Weight
# ============================================================

def calculate_pos_weight(
    labels
):

    positive = float(
        labels.sum()
    )

    negative = float(
        len(labels) - positive
    )

    if positive == 0:

        raise ValueError(
            "No positive training samples."
        )

    weight = (
        negative / positive
    )

    print(
        f"pos_weight = {weight:.4f}"
    )

    return torch.tensor(
        weight,
        dtype=torch.float32,
        device=device
    )


# ============================================================
# 8. Training
# ============================================================

def train_model():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_engineered_dataset(
        TRAIN_DATASET_PATH
    )

    labels = resolve_label_column(
        df
    )

    label_derived_cols = (
        [
            "target_failure_window",
            "failed",
        ]
        + [
            c
            for c in df.columns
            if c.startswith(
                "comp_failure"
            )
        ]
    )

    feature_cols = (
        resolve_feature_columns(
            df,
            label_derived_cols
        )
    )

    machine_id_col = None

    if "machineID" in df.columns:
        machine_id_col = "machineID"

    elif "machine_id" in df.columns:
        machine_id_col = "machine_id"

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = (
        MaintenanceSequenceDataset(
            df=df,
            feature_cols=feature_cols,
            labels=labels,
            sequence_length=SEQUENCE_LENGTH,
            step=WINDOW_STEP,
            machine_id_col=machine_id_col,
        )
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }

    if NUM_WORKERS > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = 2

    dataloader = DataLoader(
        dataset,
        **loader_kwargs
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        TabularSequenceTransformer(
            num_features=len(
                feature_cols
            )
        )
        .to(device)
    )

    # --------------------------------------------------------
    # torch.compile
    # --------------------------------------------------------

    if USE_COMPILE:

        try:

            print(
                "Compiling model with "
                "torch.compile()..."
            )

            model = torch.compile(
                model
            )

        except Exception as exc:

            print(
                "torch.compile unavailable: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pos_weight = (
        calculate_pos_weight(
            dataset.seq_labels
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_loss = float("inf")
    epochs_without_improvement = 0

    print(
        f"\nTraining on {device}"
    )

    print(
        f"AMP: {USE_AMP}"
    )

    print(
        f"Workers: {NUM_WORKERS}"
    )

    print(
        f"Windows: {len(dataset):,}"
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            dataloader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            )
        )

        for features, targets in progress:

            features = features.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Mixed precision
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                logits = model(
                    features
                )

                loss = criterion(
                    logits,
                    targets
                )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = (
            running_loss
            / max(
                len(dataloader),
                1
            )
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={epoch_loss:.4f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            epochs_without_improvement = 0

            save_trained_model(
                model,
                MODEL_SAVE_PATH
            )

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "Early stopping triggered."
                )

                break

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    feature_cols_path = (
        SAVE_DIR
        / "feature_columns.pkl"
    )

    joblib.dump(
        feature_cols,
        feature_cols_path
    )

    training_config = {
        "sequence_length":
            SEQUENCE_LENGTH,

        "window_step":
            WINDOW_STEP,

        "num_features":
            len(feature_cols),

        "feature_columns":
            feature_cols,

        "pos_weight":
            float(
                pos_weight.item()
            ),
    }

    config_path = (
        SAVE_DIR
        / "training_config.pkl"
    )

    joblib.dump(
        training_config,
        config_path
    )

    print(
        "\nTraining complete."
    )

    return model, feature_cols


# ============================================================
# 9. Save
# ============================================================

def save_trained_model(
    model,
    path
):

    # torch.compile can wrap the original model.
    # state_dict still works, but unwrap when possible.

    if hasattr(
        model,
        "_orig_mod"
    ):

        model_to_save = (
            model._orig_mod
        )

    else:

        model_to_save = model

    torch.save(
        model_to_save.state_dict(),
        path
    )

    print(
        f"Saved model: {path}"
    )


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    train_model()