import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------------------------------------
# 1. Custom Memory-Efficient Sequence Dataset
# ---------------------------------------------------------
class MaintenanceSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col='failed', seq_len=32):
        self.seq_len = seq_len
        # Store data as a single contiguous 2D float32 array (~7.3 GB instead of 218 GB)
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.num_samples = len(df) - seq_len + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        # Slice window on the fly
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

# ---------------------------------------------------------
# 2. Predictive Maintenance Sequence Model (LSTM / GRU)
# ---------------------------------------------------------
class PredictiveMaintenanceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the hidden state of the last time step
        last_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_step_out).squeeze(-1)
        return logits

# ---------------------------------------------------------
# 3. Training & Evaluation Pipeline
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_batch)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * len(y_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_pred.extend(probs)
                y_true.extend(y_batch.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        auc_score = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {auc_score:.4f}")

# ---------------------------------------------------------
# 4. Example Execution Workflow
# ---------------------------------------------------------
if __name__ == '__main__':
    # Configuration
    SEQ_LEN = 32
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Placeholder: Replace with actual dataframe loading
    # df_train = pd.read_csv('train_data.csv')
    # df_val = pd.read_csv('val_data.csv')
    
    feature_cols = ['vibration', 'temperature', 'pressure', 'rpm']  # Adjust based on dataset
    target_col = 'failed'
    
    # Mock dataframe for demonstration
    mock_df_train = pd.DataFrame({col: np.random.randn(5000) for col in feature_cols})
    mock_df_train[target_col] = np.random.choice([0, 1], size=5000, p=[0.95, 0.05])
    
    mock_df_val = pd.DataFrame({col: np.random.randn(1000) for col in feature_cols})
    mock_df_val[target_col] = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

    # Instantiate datasets & dataloaders
    train_dataset = MaintenanceSequenceDataset(mock_df_train, feature_cols, target_col, seq_len=SEQ_LEN)
    val_dataset = MaintenanceSequenceDataset(mock_df_val, feature_cols, target_col, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Initialize model, loss with class-weight handling (useful for rare failure events), and optimizer
    model = PredictiveMaintenanceModel(input_dim=len(feature_cols))
    pos_weight = torch.tensor([10.0]).to(DEVICE)  # Optional class balance weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS, device=DEVICE)