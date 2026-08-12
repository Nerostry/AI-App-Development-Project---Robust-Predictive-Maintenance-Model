import pandas as pd

# Load the combined dataset
df_combined = pd.read_csv('/content/df_combined.csv')

# 1. Drop 'machineID'
df_combined = df_combined.drop(columns=['machineID'])

# 2. Apply One-Hot Encoding to categorical columns
# Note: dummy_na=False (default) keeps NaNs as all zeros in OHE columns.
df_combined = pd.get_dummies(
    df_combined,
    columns=['model', 'errorID', 'comp_failure'],
    dtype=int,  # Outputs 1/0 instead of True/False
)

# Display the updated DataFrame shape and head
print(f"New DataFrame shape: {df_combined.shape}")
print(df_combined.head())

# Save to Colab session storage
df_combined.to_csv('model_dataset.csv', index=False)
print("Saved model_dataset.csv successfully!")