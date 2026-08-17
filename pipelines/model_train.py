import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as models

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Select computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. DATASET DEFINITION & SAMPLER
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
        self.numerical_data = np.random.randn(num_samples, 5).astype(np.float32)
        self.timeseries_data = np.random.randn(
            num_samples, sequence_length, 4
        ).astype(np.float32)
        self.image_data = np.random.randn(
            num_samples, img_size[0], img_size[1], img_size[2]
        ).astype(np.float32)
        self.labels = np.random.choice(
            [0, 1], size=num_samples, p=[1 - failure_rate, failure_rate]
        )
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
# 2. STREAMING MULTIMODAL TRANSFORMER MODEL
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
        self.ts_projection = nn.Linear(ts_features, embed_dim)
        self.num_projection = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.img_projection = nn.Linear(resnet.fc.in_features, embed_dim)

        # 2. Cross-Modal Fusion Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.transformer_fusion = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 3. Prediction Heads
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, ts, num, img):
        ts_emb = self.ts_projection(ts)
        num_emb = self.num_projection(num).unsqueeze(1)
        img_feat = self.img_encoder(img).squeeze(-1).squeeze(-1)
        img_emb = self.img_projection(img_feat).unsqueeze(1)
        
        fused_sequence = torch.cat([ts_emb, num_emb, img_emb], dim=1)
        transformer_out = self.transformer_fusion(fused_sequence)
        pooled = transformer_out.mean(dim=1)
        
        failure_logits = self.classifier_head(pooled).squeeze(-1)
        ttf_pred = F.relu(self.rul_head(pooled).squeeze(-1))
        return failure_logits, ttf_pred


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model(save_path="model.pt"):
    dataset = MultimodalMaintenanceDataset(
        num_samples=1200, sequence_length=64, failure_rate=0.08
    )
    dataloader = build_oversampled_dataloader(dataset, batch_size=32)
    model = StreamingMultimodalTransformer().to(device)

    # Class weighting loss calculation
    num_positives = sum(dataset.labels)
    num_negatives = len(dataset.labels) - num_positives
    pos_weight = torch.tensor([num_negatives / max(num_positives, 1)]).to(device)
    
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rul_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )

    model.train()
    print(f"Beginning Training on device: {device}...")
    for epoch in range(3):
        total_loss = 0.0
        for batch in dataloader:
            ts = batch["ts"].to(device)
            num = batch["num"].to(device)
            img = batch["img"].to(device)
            labels = batch["label"].to(device)
            ttf = batch["ttf"].to(device)

            optimizer.zero_grad()
            logits, ttf_pred = model(ts, num, img)
            
            cls_loss = cls_criterion(logits, labels)
            rul_loss = rul_criterion(ttf_pred, ttf)
            loss = cls_loss + 0.01 * rul_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/3 - Loss: {total_loss / len(dataloader):.4f}")

    # Save trained checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"\nModel training complete and saved to '{save_path}'")


if __name__ == "__main__":
    train_model()