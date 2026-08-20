import json
import os
import sqlite3
import cv2
import numpy as np
from kafka import KafkaConsumer
from ultralytics import YOLO

# 1. Initialize Database Schema
DB_NAME = "pipeline_results.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT,
            temperature REAL,
            pressure REAL,
            sensor_value REAL,
            calculated_index REAL,
            image_path TEXT,
            detected_objects TEXT,
            detection_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()
print("📁 Database initialized: pipeline_results.db")

# 2. Load PyTorch / YOLO Model
print("🤖 Loading YOLO Object Detection model...")
model = YOLO('yolov8n.pt')  # Downloads lightweight pre-trained model on first run

# 3. Connect to Kafka
consumer = KafkaConsumer(
    'raw-dataset-events',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='dataset-processors',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("🚀 Processor Service running with ML + DB Persistence...\n")

for message in consumer:
    event = message.value
    event_id = event.get('eventId')
    timestamp = event.get('timestamp')
    print("=" * 50)
    print(f"📥 Processing Event: {event_id}")

    # --- A. Numerical Data Analysis ---
    num_data = event.get('numericData', {})
    temp = num_data.get('temperature', 0.0)
    pressure = num_data.get('pressure', 0.0)
    sensor_val = num_data.get('sensorValue', 0.0)
    calculated_index = (temp * 0.5) + (pressure * 0.3) + (sensor_val * 0.2)

    # --- B. ML / Computer Vision Analysis ---
    image_path = event.get('imagePath')
    detected_classes = []
    
    if image_path and os.path.exists(image_path):
        results = model(image_path, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                detected_classes.append(class_name)
        
        print(f"🖼️  Image ML Inference complete -> Detected: {detected_classes}")
    else:
        print("ℹ️  No valid image file attached.")

    # --- C. Persist to Database ---
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processed_events 
        (event_id, timestamp, temperature, pressure, sensor_value, calculated_index, image_path, detected_objects, detection_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        event_id,
        timestamp,
        temp,
        pressure,
        sensor_val,
        calculated_index,
        image_path,
        json.dumps(detected_classes),
        len(detected_classes)
    ))
    conn.commit()
    conn.close()

    print(f"💾 Persisted results to Database for {event_id}\n")
