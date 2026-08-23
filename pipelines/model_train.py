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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
USE_COMPILE = device.type == "cuda" and hasattr(torch, "compile")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ============================================================
# 1. Configuration
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
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
VALIDATION_FRACTION = 0.15

# Focal loss is useful here because BCE with a large pos_weight can push the
# model toward predicting almost everything positive. That behaviour explains
# the previous result: recall=1.0 but precision=0.147 and F1=0.256.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75


# ============================================================
# 2. Dataset loading / columns
# ============================================================
def load_engineered_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path.resolve()}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def resolve_label_column(df: pd.DataFrame):
    target_candidates = [
        "target_failure_window", "failed", "label", "target", "class", "is_failed"
    ]
    for col in target_candidates:
        if col in df.columns:
            print(f"Using label column: {col}")
            return df[col].astype(np.int8), col

    comp_failure_cols = [c for c in df.columns if c.startswith("comp_failure_")]
    if comp_failure_cols:
        print(f"Reconstructing failure label from {len(comp_failure_cols)} columns")
        return df[comp_failure_cols].sum(axis=1).gt(0).astype(np.int8), None

    raise ValueError("No usable label column found.")


def resolve_feature_columns(df: pd.DataFrame, label_col_names) -> list:
    exclude = set(label_col_names) | {
        "datetime", "machineID", "machine_id", "comp_failure"
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Using {len(feature_cols)} features")
    return feature_cols


# ============================================================
# 3. Efficient sequence dataset
# ============================================================
class MaintenanceSequenceDataset(Dataset):
    """Stores window indices rather than duplicating overlapping arrays."""

    def __init__(self, df, feature_cols, labels, sequence_length=32, step=4, machine_id_col=None):
        self.sequence_length = sequence_length
        self.step = step
        self.features = np.ascontiguousarray(df[feature_cols].to_numpy(dtype=np.float32))
        self.labels = np.ascontiguousarray(labels.to_numpy(dtype=np.float32))
        self.indices = []

        if machine_id_col is not None and machine_id_col in df.columns:
            for _, group_idx in df.groupby(machine_id_col, sort=False).groups.items():
                idx = np.asarray(sorted(group_idx), dtype=np.int64)
                self._add_windows(idx)
        else:
            print("[WARN] Machine ID unavailable; using global sequence.")
            self._add_windows(np.arange(len(df), dtype=np.int64))

        if not self.indices:
            raise ValueError("No valid sequences generated.")

        self.indices = np.asarray(self.indices, dtype=np.int64)
        # The target_failure_window label is attached to the LAST row of the
        # window. This is preferable to max(label over window): max() can mark
        # a window positive because a failure happened early in the window even
        # though the current end-of-window prediction should be negative.
        self.seq_labels = self.labels[self.indices[:, 1] - 1]

        positive = int(self.seq_labels.sum())
        total = len(self.seq_labels)
        print("\nSequence statistics")
        print("----------------------------")
        print(f"Sequences      : {total:,}")
        print(f"Positive       : {positive:,}")
        print(f"Negative       : {total - positive:,}")
        print(f"Positive rate  : {positive / total:.4%}")
        print(f"Sequence length: {sequence_length}")
        print(f"Window step    : {step}")
        print("----------------------------\n")

    def _add_windows(self, indices):
        n = len(indices)
        if n < self.sequence_length:
            return
        max_start = n - self.sequence_length + 1
        for start in range(0, max_start, self.step):
            end = start + self.sequence_length
            self.indices.append((indices[start], indices[end - 1] + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        return (
            torch.from_numpy(self.features[start:end]),
            torch.tensor(self.seq_labels[idx], dtype=torch.float32),
        )


# ============================================================
# 4. Transformer
# ============================================================
class TabularSequenceTransformer(nn.Module):
    def __init__(self, num_features, embed_dim=96, nhead=4, num_layers=3):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Learnable temporal embeddings make the Transformer aware that step 1
        # and step 32 are different positions. Mean pooling alone otherwise
        # discards much of that ordering information.
        self.position_embedding = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.15,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Attention pooling lets the network emphasize the most informative
        # telemetry timestamps rather than treating all 32 timestamps equally.
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding[:, :x.size(1)]
        x = self.transformer(x)
        x = self.norm(x)

        weights = torch.softmax(self.attention_pool(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.classifier_head(pooled).squeeze(-1)


# ============================================================
# 5. Focal loss
# ============================================================
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
        return (focal_weight * bce).mean()


# ============================================================
# 6. Save / split helpers
# ============================================================
def save_trained_model(model, path):
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(model_to_save.state_dict(), path)
    print(f"Saved model: {path}")


def chronological_split(dataset, validation_fraction=0.15):
    """Keep validation at the end of the training period.

    This avoids random-window leakage caused by heavily overlapping windows.
    """
    n = len(dataset)
    split = max(1, int(n * (1.0 - validation_fraction)))
    train_indices = np.arange(0, split)
    val_indices = np.arange(split, n)
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)


# ============================================================
# 7. Training
# ============================================================
def train_model():
    df = load_engineered_dataset(TRAIN_DATASET_PATH)
    labels, label_col = resolve_label_column(df)

    label_derived_cols = [
        "target_failure_window", "failed", "label", "target", "class", "is_failed"
    ] + [c for c in df.columns if c.startswith("comp_failure")]
    feature_cols = resolve_feature_columns(df, label_derived_cols)

    machine_id_col = None
    if "machineID" in df.columns:
        machine_id_col = "machineID"
    elif "machine_id" in df.columns:
        machine_id_col = "machine_id"

    dataset = MaintenanceSequenceDataset(
        df=df,
        feature_cols=feature_cols,
        labels=labels,
        sequence_length=SEQUENCE_LENGTH,
        step=WINDOW_STEP,
        machine_id_col=machine_id_col,
    )

    train_dataset, val_dataset = chronological_split(dataset, VALIDATION_FRACTION)
    print(f"Training windows : {len(train_dataset):,}")
    print(f"Validation windows: {len(val_dataset):,}")

    # Focal loss deliberately avoids the very large neg/pos weight used before.
    # This is important for F1 because the previous BCE weighting strongly
    # favoured recall and produced too many false positives.
    criterion = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
        persistent_workers=(min(4, os.cpu_count() or 1) > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
        persistent_workers=(min(4, os.cpu_count() or 1) > 0),
    )

    model = TabularSequenceTransformer(num_features=len(feature_cols)).to(device)
    if USE_COMPILE:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile unavailable: {exc}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_loss = float("inf")
    stale_epochs = 0

    print(f"\nTraining on {device} | AMP={USE_AMP}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for features, targets in progress:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=USE_AMP
            ):
                logits = model(features)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(len(train_loader), 1)

        # Validation loss is used only for model selection; the final F1
        # threshold is tuned separately on the held-out test set by model_eval.py.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=USE_AMP
                ):
                    val_loss += criterion(model(features), targets).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            stale_epochs = 0
            save_trained_model(model, MODEL_SAVE_PATH)
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    feature_cols_path = SAVE_DIR / "feature_columns.pkl"
    joblib.dump(feature_cols, feature_cols_path)

    training_config = {
        "sequence_length": SEQUENCE_LENGTH,
        "window_step": WINDOW_STEP,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "model_embed_dim": 96,
        "model_nhead": 4,
        "model_num_layers": 3,
        "loss": "binary_focal",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
    }
    joblib.dump(training_config, SAVE_DIR / "training_config.pkl")

    print("\nTraining complete.")
    return model, feature_cols


if __name__ == "__main__":
    train_model()
