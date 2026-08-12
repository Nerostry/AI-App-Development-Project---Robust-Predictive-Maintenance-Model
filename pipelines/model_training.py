import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as models
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. TEMPORAL-AWARE RESAMPLING & DATASET
# ==========================================
class MultimodalMaintenanceDataset(Dataset):
  """Sequence-level sliding window dataset with time-series, numerical features, and images."""

  def __init__(
      self,
      num_samples=1000,
      sequence_length=64,
      img_size=(3, 64, 64),
      failure_rate=0.05,
  ):
    self.sequence_length = sequence_length
    self.num_samples = num_samples
    # Simulate tabular numerical features (e.g., machine age, static spec embeddings)
    self.numerical_data = np.random.randn(num_samples, 5).astype(np.float32)
    # Simulate time-series sensor data (e.g., volt, rotate, pressure, vibration)
    self.timeseries_data = np.random.randn(
        num_samples, sequence_length, 4
    ).astype(np.float32)
    # Simulate visual data (e.g., thermal/visual frame sequence or summary image per window)
    self.image_data = np.random.randn(
        num_samples, img_size[0], img_size[1], img_size[2]
    ).astype(np.float32)
    # Simulate highly imbalanced labels (1 = failure in window, 0 = normal)
    self.labels = np.random.choice(
        [0, 1], size=num_samples, p=[1 - failure_rate, failure_rate]
    )
    # Simulate Time-to-Failure (TTF) in hours (e.g., 0 to 72 hours window for failures)
    self.ttf_targets = np.where(
        self.labels == 1,
        np.random.uniform(1.0, 72.0, size=num_samples),
        168.0,
    ).astype(np.float32)

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    return {
        'ts': torch.tensor(self.timeseries_data[idx]),
        'num': torch.tensor(self.numerical_data[idx]),
        'img': torch.tensor(self.image_data[idx]),
        'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        'ttf': torch.tensor(self.ttf_targets[idx], dtype=torch.float32),
    }


def build_oversampled_dataloader(dataset, batch_size=32):
  """Sequence-level oversampling using WeightedRandomSampler to address class imbalance."""
  labels = dataset.labels
  class_counts = np.bincount(labels)
  class_weights = 1.0 / class_counts
  sample_weights = class_weights[labels]
  sampler = WeightedRandomSampler(
      weights=torch.DoubleTensor(sample_weights),
      num_samples=len(sample_weights),
      replacement=True,
  )
  return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# ==========================================
# 2. STREAMING MULTIMODAL TRANSFORMER (StreaMulT)
# ==========================================
class StreamingMultimodalTransformer(nn.Module):

  def __init__(
      self,
      ts_features=4,
      num_features=5,
      embed_dim=64,
      nhead=4,
      num_layers=2,
  ):
    super().__init__()
    # 1. Modality Encoders
    # Time-Series Projection
    self.ts_projection = nn.Linear(ts_features, embed_dim)
    # Tabular Numerical Projection
    self.num_projection = nn.Sequential(
        nn.Linear(num_features, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
    )
    # Vision Encoder (CNN Backing)
    resnet = models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(
        3, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Extract features before FC
    self.img_projection = nn.Linear(resnet.fc.in_features, embed_dim)

    # 2. Cross-Modal Fusion Transformer Engine
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=nhead,
        dim_feedforward=embed_dim * 2,
        batch_first=True,
    )
    self.transformer_fusion = nn.TransformerEncoder(
        encoder_layer, num_layers=num_layers
    )

    # 3. Multitask Prediction Heads
    # Task A: Binary Classification Probability Head (Failure Likelihood)
    self.classifier_head = nn.Sequential(
        nn.Linear(embed_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),  # Outputs logits
    )
    # Task B: Remaining Useful Life / Time-To-Failure Head (Regression in Hours)
    self.rul_head = nn.Sequential(
        nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)
    )

  def forward(self, ts, num, img):
    # Shape prep
    batch_size, seq_len, _ = ts.shape
    # Project Time-Series: [B, T, D]
    ts_emb = self.ts_projection(ts)
    # Project Tabular Numerical: [B, 1, D]
    num_emb = self.num_projection(num).unsqueeze(1)
    # Extract Visual Feature Embeddings: [B, 1, D]
    img_feat = self.img_encoder(img).squeeze(-1).squeeze(-1)
    img_emb = self.img_projection(img_feat).unsqueeze(1)
    # Cross-Modal Fusion Sequence Concatenation: [B, (T + 1 + 1), D]
    fused_sequence = torch.cat([ts_emb, num_emb, img_emb], dim=1)
    # Transformer Encoding
    transformer_out = self.transformer_fusion(fused_sequence)
    # Pooling sequence representation (mean pooling over temporal-modal tokens)
    pooled = transformer_out.mean(dim=1)
    # Multi-task Outputs
    failure_logits = self.classifier_head(pooled).squeeze(-1)
    ttf_pred = F.relu(
        self.rul_head(pooled).squeeze(-1)
    )  # Positive hours constraint
    return failure_logits, ttf_pred


# ==========================================
# 3. TRAINING & LOSS STRATEGY
# ==========================================
def train_model():
  # Setup dataset & sampler dataloader
  dataset = MultimodalMaintenanceDataset(
      num_samples=1200, sequence_length=64, failure_rate=0.08
  )
  dataloader = build_oversampled_dataloader(dataset, batch_size=32)
  model = StreamingMultimodalTransformer()

  # Weighted Cross Entropy strategy for class imbalance
  # Calculate pos_weight = negative_class_count / positive_class_count
  num_positives = sum(dataset.labels)
  num_negatives = len(dataset.labels) - num_positives
  pos_weight = torch.tensor([num_negatives / max(num_positives, 1)])
  cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  rul_criterion = nn.MSELoss()
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=1e-3, weight_decay=1e-4
  )

  model.train()
  print("Beginning Training...")
  for epoch in range(3):
    total_loss = 0.0
    for batch in dataloader:
      optimizer.zero_grad()
      logits, ttf_pred = model(batch["ts"], batch["num"], batch["img"])
      # Loss computation
      cls_loss = cls_criterion(logits, batch["label"])
      rul_loss = rul_criterion(ttf_pred, batch["ttf"])
      # Joint multi-task loss balance
      loss = cls_loss + 0.01 * rul_loss
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(
        f"Epoch {epoch+1}/3 - Loss: {total_loss / len(dataloader):.4f}"
    )
  return model, dataset


# ==========================================
# 4. POST-TRAINING THRESHOLD TUNING & METRICS
# ==========================================
def evaluate_and_tune(model, dataset):
  model.eval()
  val_loader = DataLoader(dataset, batch_size=64, shuffle=False)
  all_targets = []
  all_probs = []
  all_ttfs = []

  with torch.no_grad():
    for batch in val_loader:
      logits, ttf_pred = model(batch["ts"], batch["num"], batch["img"])
      probs = torch.sigmoid(logits)
      all_probs.extend(probs.numpy())
      all_targets.extend(batch["label"].numpy())
      all_ttfs.extend(ttf_pred.numpy())

  all_probs = np.array(all_probs)
  all_targets = np.array(all_targets)
  all_ttfs = np.array(all_ttfs)

  # Calculate Precision-Recall AUC
  precision, recall, thresholds = precision_recall_curve(
      all_targets, all_probs
  )
  pr_auc = auc(recall, precision)

  # Threshold Tuning: Finding the optimal decision threshold for F1-Score
  f1_scores = [
      2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)
  ]
  best_idx = np.argmax(f1_scores)
  # If best_idx falls out of range of thresholds array:
  optimal_threshold = (
      thresholds[best_idx] if best_idx < len(thresholds) else 0.5
  )
  best_f1 = f1_scores[best_idx]

  # Evaluate at optimal threshold
  predictions = (all_probs >= optimal_threshold).astype(int)
  tuned_precision = precision_score(
      all_targets, predictions, zero_division=0
  )
  tuned_recall = recall_score(all_targets, predictions, zero_division=0)

  print("\n" + "=" * 45)
  print("         EVALUATION & THRESHOLD RESULTS       ")
  print("=" * 45)
  print(f"PR-AUC (Average Precision) : {pr_auc:.4f}")
  print(f"Optimal Probability Threshold: {optimal_threshold:.4f}")
  print(f"F1-Score at Threshold      : {best_f1:.4f}")
  print(f"Precision at Threshold     : {tuned_precision:.4f}")
  print(f"Recall at Threshold        : {tuned_recall:.4f}")
  print("=" * 45)

  # Sample Output Inference Prediction
  sample_idx = 0
  print("\n[SAMPLE INFERENCE OUTPUT]")
  print(
      f"Predicted Probability of Failure : {all_probs[sample_idx] * 100:.2f}%"
  )
  print(
      "Tuned Failure Alert Binary      :"
      f" {'ALERT: FAILURE IMMINENT' if all_probs[sample_idx] >= optimal_threshold else 'NORMAL'}"
  )
  print(
      f"Predicted Time-To-Failure (TTF) : {all_ttfs[sample_idx]:.2f} Hours"
  )
  return optimal_threshold


# Execute Pipeline
if __name__ == "__main__":
  trained_model, val_dataset = train_model()
  optimal_threshold = evaluate_and_tune(trained_model, val_dataset)


# ==========================================
# 5. RANDOM VALIDATION ROW TEST
# ==========================================
# Select a random sample index from the validation dataset
random_idx = np.random.randint(0, len(val_dataset))

# Extract the sample data
sample = val_dataset[random_idx]
ts_data = sample["ts"].unsqueeze(0)  # Add batch dimension [1, seq_len, ts_features]
num_data = sample["num"].unsqueeze(0)  # Add batch dimension [1, num_features]
img_data = sample["img"].unsqueeze(0)  # Add batch dimension [1, C, H, W]
true_label = int(sample["label"].item())
true_ttf = sample["ttf"].item()

# Run inference with the trained model
trained_model.eval()
with torch.no_grad():
  logit, ttf_pred = trained_model(ts_data, num_data, img_data)
  prob = torch.sigmoid(logit).item()
  pred_ttf = ttf_pred.item()

threshold = optimal_threshold if "optimal_threshold" in locals() else 0.5
alert_status = (
    "ALERT: FAILURE IMMINENT" if prob >= threshold else "NORMAL"
)

# Display the selected input row and prediction results
print("=" * 55)
print(f"         RANDOM VALIDATION ROW TEST (Index: {random_idx})")
print("=" * 55)
print("\n--- INPUT DATA FEATURES ---")
print(f"Numerical Features (Age/Specs) : {num_data.squeeze(0).numpy().tolist()}")
print(
    "Time-Series Shape (Sensors)    :"
    f" {ts_data.squeeze(0).shape} (Last step:"
    f" {ts_data[0, -1].numpy().round(2).tolist()})"
)
print(f"Image Tensor Shape             : {img_data.squeeze(0).shape}")
print("\n--- GROUND TRUTH ---")
print(f"Actual Failure Label           : {true_label}")
print(f"Actual Time-To-Failure (TTF)   : {true_ttf:.2f} Hours")
print("\n--- MODEL PREDICTION ---")
print(f"Predicted Failure Probability  : {prob * 100:.2f}%")
print(f"Tuned Failure Alert Binary     : {alert_status}")
print(f"Predicted Time-To-Failure (TTF): {pred_ttf:.2f} Hours")
print("=" * 55)