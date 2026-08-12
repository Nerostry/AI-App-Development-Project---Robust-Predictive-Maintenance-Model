import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load the model dataset
df_model = pd.read_csv('/content/model_dataset.csv')

# Drop non-feature columns
# 'datetime' is a timestamp and not a numerical feature for training
X = df_model.drop(columns=['datetime', 'failed'])
y = df_model['failed']

# 2. Train-Test Split (using stratify due to class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Model Training
rf_model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42
)
rf_model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = rf_model.predict(X_test)

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))