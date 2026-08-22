import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Resolve Project Paths
# ==========================================
# Resolves the project root relative to this file's location (.../pipelines/..)
BASE_DIR = Path(__file__).resolve().parent.parent

# Input directory
CLEAN_DIR = BASE_DIR / "datasets" / "clean_dataset"
maint_path = CLEAN_DIR / "cleaned_maintenance_dataset.csv"
pdm_path = CLEAN_DIR / "PdM_combined_cleaned.csv"

# Output directory: datasets/model_train_dataset
OUTPUT_DIR = BASE_DIR / "datasets" / "model_train_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Load Cleaned Datasets
# ==========================================
print(f"Loading datasets from {CLEAN_DIR}...")
df_maint = pd.read_csv(maint_path)
df_pdm = pd.read_csv(pdm_path)

# PdM_combined_cleaned is the main telemetry + failure dataset
df_combined = df_pdm.copy()

# ==========================================
# 3. Preprocessing & One-Hot Encoding
# ==========================================
if 'machineID' in df_combined.columns:
    df_combined = df_combined.drop(columns=['machineID'])

# One-Hot Encoding on categorical columns
ohe_cols = [col for col in ['model', 'errorID', 'comp_failure'] if col in df_combined.columns]
if ohe_cols:
    df_combined = pd.get_dummies(
        df_combined,
        columns=ohe_cols,
        dtype=int,
    )

# ==========================================
# 4. Feature Engineering for Numerical Columns
# ==========================================
numerical_cols = ['volt', 'rotate', 'pressure', 'vibration', 'age']
numerical_cols = [c for c in numerical_cols if c in df_combined.columns]

# Ensure chronological ordering if datetime exists
if 'datetime' in df_combined.columns:
    df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])
    df_combined = df_combined.sort_values('datetime').reset_index(drop=True)

# Rolling statistics (window = 3)
window = 3
for col in numerical_cols:
    df_combined[f'{col}_roll_mean_{window}'] = (
        df_combined[col].rolling(window=window, min_periods=1).mean()
    )
    df_combined[f'{col}_roll_std_{window}'] = (
        df_combined[col].rolling(window=window, min_periods=1).std().fillna(0)
    )

# Lag features (t-1)
for col in numerical_cols:
    df_combined[f'{col}_lag_1'] = df_combined[col].shift(1).fillna(df_combined[col])

# Standard Scaling
engineered_cols = numerical_cols + [
    f'{col}_roll_mean_{window}' for col in numerical_cols
] + [
    f'{col}_roll_std_{window}' for col in numerical_cols
] + [
    f'{col}_lag_1' for col in numerical_cols
]
engineered_cols = [c for c in engineered_cols if c in df_combined.columns]

scaler = StandardScaler()
df_combined[engineered_cols] = scaler.fit_transform(df_combined[engineered_cols])

# ==========================================
# 5. Export Artifacts to model_train_dataset
# ==========================================
scaler_output_path = OUTPUT_DIR / "numerical_feature_scaler.pkl"
dataset_output_path = OUTPUT_DIR / "model_dataset.csv"

joblib.dump(scaler, scaler_output_path)
print(f"Saved fitted StandardScaler to: {scaler_output_path}")

df_combined.to_csv(dataset_output_path, index=False)
print(f"Saved processed dataset to: {dataset_output_path}")
print(f"Final dataset shape: {df_combined.shape}")