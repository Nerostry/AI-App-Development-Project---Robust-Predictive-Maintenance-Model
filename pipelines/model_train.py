import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# ============================================================
# 0. Reproducibility & Performance
# ============================================================
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ============================================================
# 1. Paths & Hyperparameters
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "model_train_dataset"
TRAIN_DATASET_PATH = DATASET_DIR / "train_scaled.csv"
SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = SAVE_DIR / "predictive_maintenance_model.pth"

SEQUENCE_LENGTH = 32
WINDOW_STEP = 4
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-4
VALIDATION_FRACTION = 0.20
EARLY_STOPPING_PATIENCE = 6

# The old neg/pos weight was about 5.8x for a 14.7% positive rate.
# That can make the model predict almost everything as positive.  A softer
# weight usually gives a better precision/recall trade-off when F1 is the goal.
POS_WEIGHT_POWER = 0.50  # sqrt(neg/pos)

# ============================================================
# 2. Load Data
# ============================================================
def load_engineered_dataset(dataset_path: Path = TRAIN_DATASET_PATH) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find engineered dataset at {dataset_path.resolve()}. "
            "Did you run the cleaning/feature-engineering script first?"
        )
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def resolve_label_column(df: pd.DataFrame) -> pd.Series:
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
            print(f"Using existing '{col}' column as binary label.")
            return df[col].astype(np.float32)

    comp_failure_cols = [c for c in df.columns if c.startswith("comp_failure_")]
    if comp_failure_cols:
        print(
            f"Reconstructing binary label from {len(comp_failure_cols)} "
            "comp_failure dummy columns."
        )
        return (df[comp_failure_cols].sum(axis=1) > 0).astype(np.float32)

    raise ValueError("Could not find a usable binary label column.")


def resolve_feature_columns(df: pd.DataFrame, label_col_names: list) -> list:
    exclude = set(label_col_names) | {
        "datetime",
        "machineID",
        "machine_id",
        "comp_failure",
    }

    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns available for training.")

    print(f"Using {len(feature_cols)} feature columns for training.")
    return feature_cols

# ============================================================
# 3. Sequence Dataset
# ============================================================
class MaintenanceSequenceDataset(Dataset):
    """Build fixed-length machine-local windows without crossing machines."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        labels: pd.Series,
        sequence_length: int = SEQUENCE_LENGTH,
        step: int = WINDOW_STEP,
        has_machine_id: bool = False,
        machine_id_col: str = "machineID",
    ):
        self.sequence_length = sequence_length
        self.step = step
        self.feature_cols = feature_cols
        self.sequences = []
        self.seq_labels = []
        self.window_machine_ids = []
        self.window_end_positions = []

        features = np.ascontiguousarray(
            df[feature_cols].to_numpy(dtype=np.float32)
        )
        label_arr = labels.to_numpy(dtype=np.float32)

        if has_machine_id and machine_id_col in df.columns:
            for machine_id, group_idx in df.groupby(machine_id_col, sort=False).groups.items():
                idx = np.asarray(sorted(group_idx), dtype=np.int64)
                self._chunk_into_windows(
                    features[idx],
                    label_arr[idx],
                    machine_id,
                )
        else:
            print("[WARN] machine ID unavailable; using one global sequence.")
            self._chunk_into_windows(features, label_arr, "__global__")

        if not self.sequences:
            raise ValueError(
                f"No sequences were built. Dataset must contain at least "
                f"{sequence_length} rows per machine."
            )

        self.sequences = np.stack(self.sequences).astype(np.float32)
        self.seq_labels = np.asarray(self.seq_labels, dtype=np.float32)
        self.window_machine_ids = np.asarray(self.window_machine_ids)

        print(
            f"Built {len(self.sequences):,} sequences of length {sequence_length}. "
            f"Positive rate: {self.seq_labels.mean():.4%}"
        )

    def _chunk_into_windows(
        self,
        feat_group: np.ndarray,
        label_group: np.ndarray,
        machine_id,
    ):
        n = len(feat_group)
        if n < self.sequence_length:
            return

        for start in range(0, n - self.sequence_length + 1, self.step):
            end = start + self.sequence_length
            self.sequences.append(feat_group[start:end])

            # Predict failure in the 24h target window attached to the FINAL
            # timestamp of the observed sequence. This prevents labels from
            # earlier timestamps inside the sequence being mixed together.
            self.seq_labels.append(label_group[end - 1])
            self.window_machine_ids.append(machine_id)
            self.window_end_positions.append(end - 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.sequences[idx]),
            "label": torch.tensor(self.seq_labels[idx], dtype=torch.float32),
        }

# ============================================================
# 4. Temporal Transformer
# ============================================================
class TabularSequenceTransformer(nn.Module):
    """Transformer with explicit temporal position information and attention pooling."""

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        max_seq_len: int = SEQUENCE_LENGTH,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.05),
        )

        # Learned positional embeddings are essential here: vanilla
        # self-attention is otherwise permutation-invariant.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Learned attention pooling lets the model focus on the most useful
        # time steps instead of averaging all 32 observations equally.
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 48),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding[:, : x.size(1)]
        x = self.transformer(x)

        attention_logits = self.attention_pool(x).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(x * attention_weights, dim=1)

        return self.classifier_head(pooled).squeeze(-1)

# ============================================================
# 5. Training Utilities
# ============================================================
def make_machine_temporal_split(dataset: MaintenanceSequenceDataset):
    """Split each machine's windows chronologically; no future windows enter training."""
    train_indices = []
    val_indices = []

    for machine_id in np.unique(dataset.window_machine_ids):
        indices = np.flatnonzero(dataset.window_machine_ids == machine_id)
        indices = indices[np.argsort(dataset.window_end_positions)[indices]]

        if len(indices) < 5:
            train_indices.extend(indices.tolist())
            continue

        split = int(len(indices) * (1.0 - VALIDATION_FRACTION))
        split = min(max(split, 1), len(indices) - 1)
        train_indices.extend(indices[:split].tolist())
        val_indices.extend(indices[split:].tolist())

    if not train_indices or not val_indices:
        raise ValueError("Unable to create a non-empty chronological validation split.")

    return np.asarray(train_indices), np.asarray(val_indices)


def calculate_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positive = float(np.sum(labels))
    negative = float(len(labels) - positive)

    if positive <= 0:
        raise ValueError("No positive training sequences found.")

    raw_ratio = negative / positive
    weight = raw_ratio ** POS_WEIGHT_POWER

    print(
        f"Class balance: positive={positive:.0f}, negative={negative:.0f}, "
        f"raw_ratio={raw_ratio:.3f}, pos_weight={weight:.3f}"
    )

    return torch.tensor(weight, dtype=torch.float32, device=device)


def save_trained_model(model: torch.nn.Module, model_path: Path | str):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model state successfully saved to '{model_path}'")


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(features)
                loss = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)

# ============================================================
# 6. Training
# ============================================================
def train_model(
    save_path: Path = MODEL_SAVE_PATH,
    dataset_path: Path = TRAIN_DATASET_PATH,
):
    df = load_engineered_dataset(dataset_path)
    labels = resolve_label_column(df)

    label_derived_cols = [
        "target_failure_window",
        "failed",
    ] + [c for c in df.columns if c.startswith("comp_failure")]

    feature_cols = resolve_feature_columns(df, label_derived_cols)

    machine_id_col = (
        "machineID"
        if "machineID" in df.columns
        else "machine_id" if "machine_id" in df.columns else None
    )
    has_machine_id = machine_id_col is not None

    dataset = MaintenanceSequenceDataset(
        df=df,
        feature_cols=feature_cols,
        labels=labels,
        sequence_length=SEQUENCE_LENGTH,
        step=WINDOW_STEP,
        has_machine_id=has_machine_id,
        machine_id_col=machine_id_col or "machineID",
    )

    train_indices, val_indices = make_machine_temporal_split(dataset)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=USE_AMP,
        num_workers=min(4, os.cpu_count() or 1),
        persistent_workers=(os.cpu_count() or 1) > 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=USE_AMP,
        num_workers=min(4, os.cpu_count() or 1),
        persistent_workers=(os.cpu_count() or 1) > 1,
    )

    model = TabularSequenceTransformer(
        num_features=len(feature_cols),
    ).to(device)

    train_labels = dataset.seq_labels[train_indices]
    pos_weight = calculate_pos_weight(train_labels)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    print(f"\nTraining on {device}")
    print(f"Training sequences   : {len(train_dataset):,}")
    print(f"Validation sequences : {len(val_dataset):,}")
    print(f"AMP                  : {USE_AMP}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_trained_model(model, save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    feature_cols_path = save_path.parent / "feature_columns.pkl"
    joblib.dump(feature_cols, feature_cols_path)

    training_config = {
        "sequence_length": SEQUENCE_LENGTH,
        "window_step": WINDOW_STEP,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "embed_dim": 96,
        "nhead": 4,
        "num_layers": 3,
        "pos_weight": float(pos_weight.item()),
        "pos_weight_power": POS_WEIGHT_POWER,
        "validation_fraction": VALIDATION_FRACTION,
        "best_validation_loss": best_val_loss,
    }
    joblib.dump(training_config, save_path.parent / "training_config.pkl")

    print(f"Saved feature column order to '{feature_cols_path}'")
    print("Training complete.")
    return model, feature_cols


if __name__ == "__main__":
    train_model()
