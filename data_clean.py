import pandas as pd


def combining_datasets():
    import pandas as pd

    # Load the datasets
    df_telemetry = pd.read_csv('/content/PdM_telemetry.csv')
    df_machines = pd.read_csv('/content/PdM_machines.csv')
    df_failures = pd.read_csv('/content/PdM_failures.csv')
    df_errors = pd.read_csv('/content/PdM_errors.csv')
    df_maint = pd.read_csv('/content/PdM_maint.csv')

    # Convert 'datetime' columns to datetime objects for merging
    df_telemetry['datetime'] = pd.to_datetime(df_telemetry['datetime'])
    df_failures['datetime'] = pd.to_datetime(df_failures['datetime'])
    df_errors['datetime'] = pd.to_datetime(df_errors['datetime'])
    df_maint['datetime'] = pd.to_datetime(df_maint['datetime'])

    print("Datasets loaded and 'datetime' columns converted.")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Combine 'comp' and 'failure' into 'comp_failure'
df_combined['comp_failure'] = df_combined['failure'].combine_first(df_combined['comp'])

# Drop redundant 'failure' and 'comp' columns
df_combined.drop(columns=['failure', 'comp'], inplace=True)

# Prepare features for KNN imputation on missing target cases
feature_cols = ['volt', 'rotate', 'pressure', 'vibration', 'model', 'age']

# Preprocess features (One-Hot Encoding & Scaling)
X_all = pd.get_dummies(df_combined[feature_cols], columns=['model'], drop_first=True)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_all), index=df_combined.index)

# Define masks for known cases vs. target cases (missing comp_failure where errorID exists)
train_mask = df_combined['comp_failure'].notna()
target_mask = df_combined['comp_failure'].isna() & df_combined['errorID'].notna()

X_train = X_scaled[train_mask]
y_train = df_combined.loc[train_mask, 'comp_failure']
X_target = X_scaled[target_mask]

# Fit KNN Classifier to impute missing component failures
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predicted_comps = knn.predict(X_target)

# Fill in the imputed values
df_combined.loc[target_mask, 'comp_failure'] = predicted_comps

# Create target indicator column 'failed'
df_combined['failed'] = df_combined['comp_failure'].notna().astype(int)

# Save processed dataframe to CSV
df_combined.to_csv('df_combined.csv', index=False)
print("Saved df_combined.csv successfully!")

df = df.drop_duplicates()