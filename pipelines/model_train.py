import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "model_train_dataset"
TRAIN_DATASET_PATH = DATASET_DIR / "train_scaled.csv"
SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = SAVE_DIR / "predictive_maintenance_model.pth"

SEQUENCE_LENGTH = 32
WINDOW_STEP = 16
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
VALIDATION_FRACTION = 0.15
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
GRADIENT_CLIP_NORM = 1.0


def load_engineered_dataset(dataset_path: Path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path.resolve()}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def resolve_label_column(df):
    target_candidates = ["target_failure_window", "failed", "label", "target", "class", "is_failed"]
    for col in target_candidates:
        if col in df.columns:
            print(f"Using label column: {col}")
            return df[col].astype(np.int8), col
    comp_failure_cols = [c for c in df.columns if c.startswith("comp_failure_")]
    if comp_failure_cols:
        print(f"Reconstructing failure label from {len(comp_failure_cols)} columns")
        return df[comp_failure_cols].sum(axis=1).gt(0).astype(np.int8), None
    raise ValueError("No usable label column found.")


def resolve_feature_columns(df, label_col_names):
    exclude = set(label_col_names) | {"datetime", "machineID", "machine_id", "comp_failure"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    return feature_cols


def resolve_modalities(feature_cols):
    event_cols = [c for c in feature_cols if c.startswith("err_") or c.startswith("maint_")]
    static_cols = [c for c in feature_cols if c == "age" or c.startswith("model_")]
    telemetry_cols = [c for c in feature_cols if c not in set(event_cols) and c not in set(static_cols)]
    if not telemetry_cols:
        raise ValueError("No telemetry features remain after modality split.")
    print("\nModalities")
    print("----------------------------")
    print(f"Telemetry features : {len(telemetry_cols):,}")
    print(f"Event features     : {len(event_cols):,}")
    print(f"Static features    : {len(static_cols):,}")
    print("----------------------------\n")
    return telemetry_cols, event_cols, static_cols


class MaintenanceMultimodalDataset(Dataset):
    """Aligned telemetry, event and static views for each machine window."""
    def __init__(self, df, telemetry_cols, event_cols, static_cols, labels, sequence_length=32, step=16, machine_id_col=None):
        self.sequence_length = sequence_length
        self.step = step
        self.telemetry = np.ascontiguousarray(df[telemetry_cols].to_numpy(dtype=np.float32))
        self.events = np.ascontiguousarray(df[event_cols].to_numpy(dtype=np.float32) if event_cols else np.zeros((len(df), 1), dtype=np.float32))
        self.static = np.ascontiguousarray(df[static_cols].to_numpy(dtype=np.float32) if static_cols else np.zeros((len(df), 1), dtype=np.float32))
        self.labels = np.ascontiguousarray(labels.to_numpy(dtype=np.float32))
        self.indices = []

        if machine_id_col is not None and machine_id_col in df.columns:
            groups = df.groupby(machine_id_col, sort=False).groups
            for _, group_idx in groups.items():
                self._add_windows(np.asarray(sorted(group_idx), dtype=np.int64))
        else:
            print("[WARN] Machine ID unavailable. Using global sequence.")
            self._add_windows(np.arange(len(df), dtype=np.int64))

        if not self.indices:
            raise ValueError("No valid sequences generated.")
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.seq_labels = self.labels[self.indices[:, 1] - 1]

        positive = int(self.seq_labels.sum())
        total = len(self.seq_labels)
        print("\nMultimodal sequence statistics")
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
            torch.from_numpy(self.telemetry[start:end]),
            torch.from_numpy(self.events[start:end]),
            torch.from_numpy(self.static[end - 1]),
            torch.tensor(self.seq_labels[idx], dtype=torch.float32),
        )


# Compatibility alias for older imports. The returned sample is now multimodal.
MaintenanceSequenceDataset = MaintenanceMultimodalDataset


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=96, nhead=4, num_layers=3, dropout=0.15):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=dropout, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Sequential(nn.Linear(embed_dim, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding[:, :x.size(1)]
        x = self.norm(self.transformer(x))
        weights = torch.softmax(self.attention_pool(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class StaticMetadataEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Dropout(0.15), nn.Linear(64, output_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.network(x)


class MultiModalPredictiveMaintenanceModel(nn.Module):
    """Late fusion of time-series telemetry, event history and static machine metadata."""
    def __init__(self, telemetry_dim, event_dim, static_dim, embed_dim=96, nhead=4, num_layers=3):
        super().__init__()
        self.telemetry_encoder = TemporalTransformerEncoder(telemetry_dim, embed_dim, nhead, num_layers)
        self.event_encoder = TemporalTransformerEncoder(event_dim, embed_dim // 2, nhead=4, num_layers=2)
        self.static_encoder = StaticMetadataEncoder(static_dim, output_dim=64)
        fusion_dim = embed_dim + embed_dim // 2 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Dropout(0.30), nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(0.20), nn.Linear(64, 1),
        )

    def forward(self, telemetry, events, static):
        telemetry_repr = self.telemetry_encoder(telemetry)
        event_repr = self.event_encoder(events)
        static_repr = self.static_encoder(static)
        return self.fusion(torch.cat([telemetry_repr, event_repr, static_repr], dim=-1)).squeeze(-1)


# New name is preferred; old name remains an alias for import compatibility.
TabularSequenceTransformer = MultiModalPredictiveMaintenanceModel


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


def chronological_split(dataset, validation_fraction=0.15):
    n = len(dataset)
    if n < 2:
        raise ValueError("Need at least two sequences for validation.")
    split = min(max(int(n * (1.0 - validation_fraction)), 1), n - 1)
    return (
        torch.utils.data.Subset(dataset, np.arange(0, split, dtype=np.int64)),
        torch.utils.data.Subset(dataset, np.arange(split, n, dtype=np.int64)),
    )


def save_trained_model(model, path):
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(model_to_save.state_dict(), path)
    print(f"Saved model: {path}")


def make_dataloader(dataset, shuffle):
    num_workers = min(4, os.cpu_count() or 1)
    kwargs = {
        "batch_size": BATCH_SIZE, "shuffle": shuffle, "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def train_model():
    df = load_engineered_dataset(TRAIN_DATASET_PATH)
    labels, _ = resolve_label_column(df)
    label_derived_cols = ["target_failure_window", "failed", "label", "target", "class", "is_failed", "comp_failure"]
    label_derived_cols += [c for c in df.columns if c.startswith("comp_failure")]
    feature_cols = resolve_feature_columns(df, label_derived_cols)
    telemetry_cols, event_cols, static_cols = resolve_modalities(feature_cols)

    machine_id_col = "machineID" if "machineID" in df.columns else ("machine_id" if "machine_id" in df.columns else None)
    dataset = MaintenanceMultimodalDataset(
        df=df, telemetry_cols=telemetry_cols, event_cols=event_cols,
        static_cols=static_cols, labels=labels, sequence_length=SEQUENCE_LENGTH,
        step=WINDOW_STEP, machine_id_col=machine_id_col,
    )
    train_dataset, validation_dataset = chronological_split(dataset, VALIDATION_FRACTION)
    print(f"Training windows  : {len(train_dataset):,}")
    print(f"Validation windows: {len(validation_dataset):,}")

    train_loader = make_dataloader(train_dataset, True)
    validation_loader = make_dataloader(validation_dataset, False)
    model = MultiModalPredictiveMaintenanceModel(
        telemetry_dim=len(telemetry_cols), event_dim=max(len(event_cols), 1),
        static_dim=max(len(static_cols), 1), embed_dim=96, nhead=4, num_layers=3,
    ).to(device)

    if USE_COMPILE:
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile unavailable: {exc}")

    criterion = BinaryFocalLoss(FOCAL_ALPHA, FOCAL_GAMMA).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    print(f"\nTraining multimodal model on {device}")
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for telemetry, events, static, targets in progress:
            telemetry = telemetry.to(device, non_blocking=True)
            events = events.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                loss = criterion(model(telemetry, events, static), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_train_loss / max(len(train_loader), 1)
        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad():
            for telemetry, events, static, targets in validation_loader:
                telemetry = telemetry.to(device, non_blocking=True)
                events = events.to(device, non_blocking=True)
                static = static.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                    validation_loss = criterion(model(telemetry, events, static), targets)
                running_validation_loss += validation_loss.item()

        validation_loss = running_validation_loss / max(len(validation_loader), 1)
        scheduler.step(validation_loss)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_trained_model(model, MODEL_SAVE_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    joblib.dump(feature_cols, SAVE_DIR / "feature_columns.pkl")
    training_config = {
        "model_type": "multimodal_late_fusion_transformer",
        "sequence_length": SEQUENCE_LENGTH,
        "window_step": WINDOW_STEP,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "telemetry_columns": telemetry_cols,
        "event_columns": event_cols,
        "static_columns": static_cols,
        "telemetry_dim": len(telemetry_cols),
        "event_dim": max(len(event_cols), 1),
        "static_dim": max(len(static_cols), 1),
        "model_embed_dim": 96,
        "model_nhead": 4,
        "model_num_layers": 3,
        "loss": "binary_focal",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "validation_fraction": VALIDATION_FRACTION,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
    }
    joblib.dump(training_config, SAVE_DIR / "training_config.pkl")
    print("\nMultimodal training complete.")
    return model, feature_cols


if __name__ == "__main__":
    train_model()
