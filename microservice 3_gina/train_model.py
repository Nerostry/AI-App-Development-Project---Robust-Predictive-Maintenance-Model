import sqlite3
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Pull processed dataset directly from Microservice Database
print("📥 Fetching processed features from database...")
conn = sqlite3.connect("pipeline_results.db")

query = """
SELECT 
    temperature, 
    pressure, 
    sensor_value, 
    calculated_index, 
    detection_count
FROM processed_events
"""

df = pd.read_sql_query(query, conn)
conn.close()

print(f"📊 Dataset loaded: {len(df)} samples retrieved.\n")

if len(df) < 5:
    print("⚠️  Need more data to train! Run your CSV ingest script to populate more events.")
    exit(0)

# 2. Example ML Target: Predictive Maintenance Anomaly Rule
# Anomaly if temperature > 40 OR detection_count > 0 (adjust to your project criteria)
df['is_anomaly'] = ((df['temperature'] > 40) | (df['detection_count'] > 0)).astype(int)

# 3. Define Features (X) and Target (y)
X = df[['temperature', 'pressure', 'sensor_value', 'calculated_index', 'detection_count']]
y = df['is_anomaly']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
print("🤖 Training Predictive Maintenance Classifier...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test)
print("\n📋 Model Evaluation Metrics:")
print(classification_report(y_test, y_pred, zero_division=0))

print("✅ Training complete! Model is ready for deployment in microservices.")
