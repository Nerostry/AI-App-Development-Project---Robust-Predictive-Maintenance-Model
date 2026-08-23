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

# 16 = 50% overlap while avoiding the much larger number of
# near-duplicate windows produced by a step of 1 or 4.
WINDOW_STEP = 16

BATCH_SIZE = 256
EPOCHS = 30

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

VALIDATION_FRACTION = 0.15

# Focal loss replaces the previous aggressive BCE neg/pos weighting.
# A lower alpha prevents the positive class from being over-emphasised,
# which is appropriate for the observed false-positive-heavy baseline.
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

GRADIENT_CLIP_NORM = 1.0


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
                .astype(np.int8),
                col,
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
            .astype(np.int8),
            None,
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

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found."
        )

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

    The label is taken from the END of each sequence.
    This makes the training objective:

        past 32 hours of telemetry
            -> imminent/current end-of-window failure label

    rather than marking a sequence positive merely because
    a failure occurred somewhere earlier inside the window.
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

        self.features = np.ascontiguousarray(
            df[feature_cols]
            .to_numpy(dtype=np.float32)
        )

        self.labels = np.ascontiguousarray(
            labels.to_numpy(dtype=np.float32)
        )

        self.indices = []

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

                self._add_windows(idx)

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

        # IMPORTANT:
        # Use the label at the final timestep rather than
        # max(label over the whole sequence).
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
        embed_dim=96,
        nhead=4,
        num_layers=3,
    ):

        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                num_features,
                embed_dim
            ),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Learnable temporal embeddings make timestep position explicit.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                SEQUENCE_LENGTH,
                embed_dim
            )
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                dropout=0.15,
                activation="gelu",
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        # Learned attention pooling replaces mean pooling so the model can
        # focus on the timestamps most relevant to an imminent failure.
        self.attention_pool = nn.Sequential(
            nn.Linear(
                embed_dim,
                32
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                1
            ),
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(
                embed_dim,
                64
            ),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(
                64,
                1
            ),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = (
            x
            + self.position_embedding[
                :, :x.size(1)
            ]
        )

        x = self.transformer(x)

        x = self.norm(x)

        attention_scores = (
            self.attention_pool(x)
            .squeeze(-1)
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        pooled = torch.sum(
            x
            * attention_weights.unsqueeze(-1),
            dim=1
        )

        return (
            self.classifier_head(pooled)
            .squeeze(-1)
        )


# ============================================================
# 7. Focal Loss
# ============================================================

class BinaryFocalLoss(nn.Module):

    """
    Binary focal loss.

    Unlike BCEWithLogitsLoss(pos_weight=negative/positive), this does not
    apply the extremely large class weight that previously drove recall to
    1.0 while precision collapsed to 0.147.

    alpha=0.25 deliberately avoids over-weighting positive predictions.
    gamma=2.0 down-weights easy examples and concentrates learning on
    difficult examples, including hard negatives.
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        bce = (
            nn.functional
            .binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none"
            )
        )

        probabilities = torch.sigmoid(
            logits
        )

        p_t = (
            probabilities * targets
            + (1.0 - probabilities)
            * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha)
            * (1.0 - targets)
        )

        focal_weight = (
            alpha_t
            * (1.0 - p_t).pow(
                self.gamma
            )
        )

        return (
            focal_weight * bce
        ).mean()


# ============================================================
# 8. Validation Split
# ============================================================

def chronological_split(
    dataset,
    validation_fraction=0.15
):

    """
    Hold out the final portion of the generated training windows.

    The dataset builds windows machine-by-machine in source order, so this
    split preserves ordering rather than randomly distributing overlapping
    windows between train and validation.

    The validation set is used for model selection only. It is NOT used for
    final test threshold selection.
    """

    n = len(dataset)

    if n < 2:
        raise ValueError(
            "Need at least two sequences for validation."
        )

    split = int(
        n * (1.0 - validation_fraction)
    )

    split = min(
        max(split, 1),
        n - 1
    )

    train_indices = np.arange(
        0,
        split,
        dtype=np.int64
    )

    validation_indices = np.arange(
        split,
        n,
        dtype=np.int64
    )

    return (
        torch.utils.data.Subset(
            dataset,
            train_indices
        ),
        torch.utils.data.Subset(
            dataset,
            validation_indices
        )
    )


# ============================================================
# 9. Save
# ============================================================

def save_trained_model(
    model,
    path
):

    # torch.compile wraps the original module. Save the unwrapped model so
    # model_eval.py can instantiate the same architecture and load the state.
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
# 10. DataLoader helper
# ============================================================

def make_dataloader(
    dataset,
    shuffle
):

    num_workers = min(
        4,
        os.cpu_count() or 1
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        dataset,
        **loader_kwargs
    )


# ============================================================
# 11. Training
# ============================================================

def train_model():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_engineered_dataset(
        TRAIN_DATASET_PATH
    )

    labels, label_col = (
        resolve_label_column(df)
    )

    label_derived_cols = [
        "target_failure_window",
        "failed",
        "label",
        "target",
        "class",
        "is_failed",
    ] + [
        c
        for c in df.columns
        if c.startswith(
            "comp_failure"
        )
    ]

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
    # Chronological validation split
    # --------------------------------------------------------

    train_dataset, validation_dataset = (
        chronological_split(
            dataset,
            VALIDATION_FRACTION
        )
    )

    print(
        f"Training windows : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation windows: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = make_dataloader(
        train_dataset,
        shuffle=True
    )

    validation_loader = make_dataloader(
        validation_dataset,
        shuffle=False
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

    criterion = BinaryFocalLoss(
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA
    ).to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
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

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    print(
        f"\nTraining on {device}"
    )

    print(
        f"AMP: {USE_AMP}"
    )

    print(
        f"Workers: "
        f"{min(4, os.cpu_count() or 1)}"
    )

    print(
        f"Windows: {len(dataset):,}"
    )

    for epoch in range(EPOCHS):

        # ====================================================
        # Training phase
        # ====================================================

        model.train()

        running_train_loss = 0.0

        progress = tqdm(
            train_loader,
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

            scaler.scale(
                loss
            ).backward()

            # Unscale before gradient clipping so the clip threshold is in
            # the same units as the actual gradients.
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            running_train_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_train_loss
            / max(
                len(train_loader),
                1
            )
        )

        # ====================================================
        # Validation phase
        # ====================================================

        model.eval()

        running_validation_loss = 0.0

        with torch.no_grad():

            for features, targets in (
                validation_loader
            ):

                features = features.to(
                    device,
                    non_blocking=True
                )

                targets = targets.to(
                    device,
                    non_blocking=True
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=USE_AMP
                ):

                    logits = model(
                        features
                    )

                    validation_loss = (
                        criterion(
                            logits,
                            targets
                        )
                    )

                running_validation_loss += (
                    validation_loss.item()
                )

        validation_loss = (
            running_validation_loss
            / max(
                len(validation_loader),
                1
            )
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"lr={current_lr:.2e}"
        )

        # ====================================================
        # Model selection / early stopping
        # ====================================================

        if validation_loss < (
            best_validation_loss - 1e-5
        ):

            best_validation_loss = (
                validation_loss
            )

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
    # Save feature metadata
    # --------------------------------------------------------

    feature_cols_path = (
        SAVE_DIR
        / "feature_columns.pkl"
    )

    joblib.dump(
        feature_cols,
        feature_cols_path
    )

    # --------------------------------------------------------
    # Save architecture/training metadata
    # --------------------------------------------------------

    training_config = {
        "sequence_length":
            SEQUENCE_LENGTH,

        "window_step":
            WINDOW_STEP,

        "num_features":
            len(feature_cols),

        "feature_columns":
            feature_cols,

        "model_embed_dim":
            96,

        "model_nhead":
            4,

        "model_num_layers":
            3,

        "loss":
            "binary_focal",

        "focal_alpha":
            FOCAL_ALPHA,

        "focal_gamma":
            FOCAL_GAMMA,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "validation_fraction":
            VALIDATION_FRACTION,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,
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
# 12. Main
# ============================================================

if __name__ == "__main__":

    train_model()