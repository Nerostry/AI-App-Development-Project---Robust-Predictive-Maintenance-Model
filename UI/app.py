import streamlit as st
import pandas as pd
import requests
import os
import requests
import streamlit as st

MODEL_SERVICE_HOST = os.getenv("MODEL_SERVICE_HOST", "localhost")
MODEL_SERVICE_URL = f"http://{MODEL_SERVICE_HOST}:8000/predict"

st.set_page_config(page_title="Turbofan Predictive Maintenance", layout="wide")

st.title("🛠️ Turbofan Engine Health & Predictive Maintenance")
st.write("Upload NASA Turbofan engine sensor data to generate real-time maintenance predictions.")

# 1. File Upload Component
uploaded_file = st.sidebar.file_uploader("Upload Sensor CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Ingested Sensor Data")
    st.dataframe(df.head())
    
    if st.button("Run Prediction via Microservice 1"):
        with st.spinner("Communicating with Model Microservice..."):
            payload = {"data": df.to_dict(orient="records")}
            
            try:
                response = requests.post(MODEL_SERVICE_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    results = response.json()
                    st.success("Inference Complete!")
                    st.subheader("📊 Maintenance Assessment")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted RUL (Cycles)", results.get("rul", "N/A"))
                    col2.metric("Failure Risk Level", results.get("risk_level", "N/A"))
                    col3.metric("Engine Health Index", f"{results.get('health_score', 100)}%")
                    
                    st.subheader("📉 Sensor Degradation Trend")
                    sensor_cols = [col for col in df.columns if 'sensor' in col.lower() or 'setting' in col.lower()]
                    if sensor_cols:
                        st.line_chart(df[sensor_cols[:5]])
                else:
                    st.error(f"Microservice 1 Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach Microservice 1: {e}")
else:
    st.info("👈 Please upload a CSV file containing engine sensor logs to proceed.")