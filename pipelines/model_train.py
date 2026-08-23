"""
pipelines/model_train.py

Retrained against the actual feature-engineered dataset produced by
pipelines/feature_engineer.py:

    datasets/model_train_dataset/model_dataset.csv
    datasets/model_train_dataset/numerical_feature_scaler.pkl
"""
    
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ==========================================
# 0. Reproducibility & Device
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "model_train_dataset"
DATASET_PATH = DATASET_DIR / "model_dataset.csv"
SCALER_PATH = DATASET_DIR / "numerical_feature_scaler.pkl"

SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = SAVE_DIR / "predictive_maintenance_model.pth"

SEQUENCE_LENGTH = 32   # window size (time steps per sample)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3


# ==========================================
# 2. Load Data & Scaler
# ==========================================
def load_engineered_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find engineered dataset at {path.resolve()}. "
            f"Run pipelines/feature_engineer.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded engineered dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def resolve_label_column(df: pd.DataFrame) -> pd.Series:
    """Returns a binary failure label Series, handling either schema case
    described in the module docstring."""
    if "failed" in df.columns:
        print("Using existing 'failed' column as binary label.")
        return df["failed"].astype(int)

    comp_failure_cols = [c for c in df.columns if c.startswith("comp_failure_")]
    if comp_failure_cols:
        print(
            f"'failed' column not found. Reconstructing binary label from "
            f"{len(comp_failure_cols)} comp_failure_* dummy columns."
        )
        return (df[comp_failure_cols].sum(axis=1) > 0).astype(int)

    raise ValueError(
        "Could not find a usable label column ('failed' or 'comp_failure_*' "
        "dummies). Inspect model_dataset.csv columns and update "
        "resolve_label_column() accordingly."
    )


def resolve_feature_columns(df: pd.DataFrame, label_col_names: list) -> list:
    """All numeric columns except identifiers, timestamps, and label-derived
    columns are treated as model input features."""
    exclude = set(label_col_names) | {"datetime", "machineID"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Using {len(feature_cols)} feature columns for training.")
    return feature_cols


# ==========================================
# 3. Sequence Dataset (per-machine windows)
# ==========================================
class MaintenanceSequenceDataset(Dataset):
    """Builds fixed-length sliding-window sequences from the engineered
    tabular dataset. Sequences respect machine boundaries if 'machineID'
    is present in the raw dataframe; otherwise the whole dataset is
    chunked as one continuous series (less correct, but a safe fallback)."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        labels: pd.Series,
        sequence_length: int = SEQUENCE_LENGTH,
        has_machine_id: bool = False,
    ):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.sequences = []
        self.seq_labels = []

        features = df[feature_cols].to_numpy(dtype=np.float32)
        label_arr = labels.to_numpy(dtype=np.float32)

        if has_machine_id and "machineID" in df.columns:
            for _, group_idx in df.groupby("machineID").groups.items():
                idx = np.array(sorted(group_idx))
                self._chunk_into_windows(features[idx], label_arr[idx])
        else:
            print(
                "[WARN] 'machineID' not found — building sequences over the "
                "whole dataset without respecting machine boundaries."
            )
            self._chunk_into_windows(features, label_arr)

        if len(self.sequences) == 0:
            raise ValueError(
                "No sequences were built. Check that the dataset has at "
                "least `sequence_length` rows per machine (or overall)."
            )

        self.sequences = np.stack(self.sequences)          # (N, T, F)
        self.seq_labels = np.array(self.seq_labels, dtype=np.float32)  # (N,)

        print(
            f"Built {len(self.sequences)} sequences of length "
            f"{sequence_length}. Positive sequence rate: "
            f"{self.seq_labels.mean():.4f}"
        )

    def _chunk_into_windows(self, feat_group: np.ndarray, label_group: np.ndarray):
        n = len(feat_group)
        if n < self.sequence_length:
            return  # not enough rows to form a window
        for start in range(0, n - self.sequence_length + 1, self.sequence_length):
            end = start + self.sequence_length
            window_feats = feat_group[start:end]
            window_label = label_group[start:end].max()  # positive if any failure in window
            self.sequences.append(window_feats)
            self.seq_labels.append(window_label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.sequences[idx]),
            "label": torch.tensor(self.seq_labels[idx], dtype=torch.float32),
        }


def build_oversampled_dataloader(dataset: MaintenanceSequenceDataset, batch_size: int = BATCH_SIZE):
    labels = dataset.seq_labels.astype(int)
    class_counts = np.bincount(labels, minlength=2)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid div-by-zero
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# ==========================================
# 4. Model: Transformer Encoder over Tabular Feature Sequences
# ==========================================
class TabularSequenceTransformer(nn.Module):
    """Encodes a window of engineered tabular features (rolling stats, lags,
    one-hot dummies, raw sensor readings) as a sequence and predicts whether
    the window contains an impending failure."""

    def __init__(self, num_features: int, embed_dim: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        emb = self.input_projection(x)
        encoded = self.transformer(emb)
        pooled = encoded.mean(dim=1)  # global average pooling over time steps
        logits = self.classifier_head(pooled).squeeze(-1)
        return logits


# ==========================================
# 5. Training Loop
# ==========================================
def train_model(save_path: Path = MODEL_SAVE_PATH):
    df = load_engineered_dataset()
    labels = resolve_label_column(df)
    label_derived_cols = ["failed"] + [c for c in df.columns if c.startswith("comp_failure_")]
    feature_cols = resolve_feature_columns(df, label_derived_cols)

    has_machine_id = "machineID" in df.columns
    dataset = MaintenanceSequenceDataset(
        df=df,
        feature_cols=feature_cols,
        labels=labels,
        sequence_length=SEQUENCE_LENGTH,
        has_machine_id=has_machine_id,
    )
    dataloader = build_oversampled_dataloader(dataset, batch_size=BATCH_SIZE)

    model = TabularSequenceTransformer(num_features=len(feature_cols)).to(device)

    # Class-weighted loss (in addition to oversampling, for extra imbalance help)
    pos = dataset.seq_labels.sum()
    neg = len(dataset.seq_labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    model.train()
    print(f"Beginning training on device: {device}...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            features = batch["features"].to(device)
            labels_batch = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel training complete and saved to '{save_path}'")

    # Persist feature column order — required at inference time to rebuild
    # windows in the same column order the model was trained on.
    feature_cols_path = save_path.parent / "feature_columns.pkl"
    joblib.dump(feature_cols, feature_cols_path)
    print(f"Saved feature column order to '{feature_cols_path}'")

    return model, feature_cols


if __name__ == "__main__":
    train_model()