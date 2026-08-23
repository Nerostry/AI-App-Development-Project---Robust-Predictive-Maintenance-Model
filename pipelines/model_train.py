import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 0. Reproducibility & Device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Paths & Hyperparameters
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "model_train_dataset"
TRAIN_DATASET_PATH = DATASET_DIR / "train_scaled.csv"
TEST_DATASET_PATH = DATASET_DIR / "test_scaled.csv"
SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = SAVE_DIR / "predictive_maintenance_model.pth"

SEQUENCE_LENGTH = 32
WINDOW_STEP = 4        # Stride of 4 skips duplicate adjacent windows and accelerates training by ~4x
BATCH_SIZE = 256       # Increased batch size significantly speeds up CPU/GPU throughput
EPOCHS = 15
LEARNING_RATE = 1e-3 

# 2. Load Data
def load_engineered_dataset(dataset_path: Path = TRAIN_DATASET_PATH) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find engineered dataset at {dataset_path.resolve()}. "
            "Did you run the cleaning/feature-engineering script first?"
        )
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def resolve_label_column(df: pd.DataFrame) -> pd.Series:
    target_candidates = ["target_failure_window", "failed", "label", "target", "class", "is_failed"]
    for col in target_candidates:
        if col in df.columns:
            print(f"Using existing '{col}' column as binary label.")
            return df[col].astype(int)
    comp_failure_cols = [c for c in df.columns if c.startswith("comp_failure_")]
    if comp_failure_cols:
        print(f"Reconstructing binary label from {len(comp_failure_cols)} comp_failure dummy columns.")
        return (df[comp_failure_cols].sum(axis=1) > 0).astype(int)
    raise ValueError(f"Could not find a usable label column. Available columns: {list(df.columns[:10])}...")

def resolve_feature_columns(df: pd.DataFrame, label_col_names: list) -> list:
    exclude = set(label_col_names) | {"datetime", "machineID", "machine_id", "comp_failure"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Using {len(feature_cols)} feature columns for training.")
    return feature_cols

# 3. Sequence Dataset
class MaintenanceSequenceDataset(Dataset):
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

        features = df[feature_cols].to_numpy(dtype=np.float32)
        label_arr = labels.to_numpy(dtype=np.float32)

        if has_machine_id and machine_id_col in df.columns:
            for _, group_idx in df.groupby(machine_id_col).groups.items():
                idx = np.array(sorted(group_idx))
                self._chunk_into_windows(features[idx], label_arr[idx])
        else:
            print("[WARN] machine ID column not found; building sequences over the entire dataset.")
            self._chunk_into_windows(features, label_arr)

        if len(self.sequences) == 0:
            raise ValueError(f"No sequences were built. Check that dataset has >= {sequence_length} rows per machine.")

        self.sequences = np.stack(self.sequences)
        self.seq_labels = np.array(self.seq_labels, dtype=np.float32)
        print(f"Built {len(self.sequences)} sequences of length {sequence_length}. Positive rate: {self.seq_labels.mean():.4f}")

    def _chunk_into_windows(self, feat_group: np.ndarray, label_group: np.ndarray):
        n = len(feat_group)
        if n < self.sequence_length:
            return
        for start in range(0, n - self.sequence_length + 1, self.step):
            end = start + self.sequence_length
            self.sequences.append(feat_group[start:end])
            self.seq_labels.append(label_group[start:end].max())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.sequences[idx]),
            "label": torch.tensor(self.seq_labels[idx], dtype=torch.float32),
        }

# 4. Model Architecture
class TabularSequenceTransformer(nn.Module):
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
        emb = self.input_projection(x)
        encoded = self.transformer(emb)
        pooled = encoded.mean(dim=1)
        logits = self.classifier_head(pooled).squeeze(-1)
        return logits

# 5. Training Loop
def save_trained_model(model: torch.nn.Module, model_path: Path | str):
    if isinstance(model_path, str):
        model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model state successfully saved to '{model_path}'")

def train_model(save_path: Path = MODEL_SAVE_PATH, dataset_path: Path = TRAIN_DATASET_PATH):
    df = load_engineered_dataset(dataset_path=dataset_path)
    labels = resolve_label_column(df)
    label_derived_cols = ["target_failure_window", "failed"] + [c for c in df.columns if c.startswith("comp_failure")]
    feature_cols = resolve_feature_columns(df, label_derived_cols)

    machine_id_col = "machineID" if "machineID" in df.columns else "machine_id"
    has_machine_id = machine_id_col in df.columns

    dataset = MaintenanceSequenceDataset(
        df=df,
        feature_cols=feature_cols,
        labels=labels,
        sequence_length=SEQUENCE_LENGTH,
        step=WINDOW_STEP,
        has_machine_id=has_machine_id,
        machine_id_col=machine_id_col,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0,
    )

    model = TabularSequenceTransformer(num_features=len(feature_cols)).to(device)

    pos = dataset.seq_labels.sum()
    neg = len(dataset.seq_labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    model.train()
    print(f"Beginning training on device: {device}...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]", leave=True)
        for batch in loop:
            features = batch["features"].to(device)
            labels_batch = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} Finished | Avg Loss: {avg_loss:.4f}")

    save_trained_model(model, save_path)
    feature_cols_path = save_path.parent / "feature_columns.pkl"
    joblib.dump(feature_cols, feature_cols_path)
    print(f"Saved feature column order to '{feature_cols_path}'")
    return model, feature_cols

if __name__ == "__main__":
    train_model()