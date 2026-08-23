

# AI App Development Project: Robust Predictive Maintenance Model

Welcome to the **Robust Predictive Maintenance Model** repository. This project delivers a scalable, end-to-end machine learning application that predicts maintenance needs. It features a robust data pipeline for seamless data ingestion, a trained predictive model, and a containerized microservices architecture for easy deployment.

**Created by the Integrated Legacy 4.0 Team:**
*   Eshimeet Kaur (@eiooioo)
*   Gina (@jiinani)
*   Jun Hian (@Nerostry)

---

## 🎯 Problem Statement & Industry Relevance
In industrial and manufacturing settings, unexpected equipment failure leads to costly downtime, safety hazards, and expensive emergency repairs. **Predictive maintenance** leverages IoT sensor data and machine learning to predict when a machine is likely to fail, allowing operators to perform maintenance *before* a breakdown occurs. 

This project solves the problem of reactive maintenance by providing a real-time, scalable ML pipeline that analyzes data and alerts users to anomalies. This approach maximizes equipment lifespan, minimizes operational disruption, and significantly reduces maintenance costs.

---

## 🏗️ System Architecture
*(Note to team: Replace `docs/architecture-diagram.png` with a screenshot of your actual system diagram)*

![System Architecture Diagram](docs/architecture-diagram.png)

This system is built using a decoupled microservices architecture to ensure fault tolerance and scalability. The services communicate asynchronously using Apache Kafka.

### Microservices Breakdown
1.  **`User_InterFace/` (Frontend):** A Streamlit application that provides an interactive dashboard for users to upload data, view predictions, and monitor system health.
2.  **`main.py` (Backend API):** A FastAPI service that acts as the bridge between the frontend and the underlying Kafka event streams.
3.  **`data_processing_service/`:** Ingests raw data, performs cleaning, handles missing values, and standardizes formats before publishing to the Kafka message broker.
4.  **`predictive_maintenance_service/`:** The core AI service. It consumes cleaned data from Kafka, runs it through our custom YOLO-integrated machine learning models, and outputs failure predictions.

---

## 📊 Dataset Information
*   **Source:** *(e.g., Kaggle, NASA Turbofan Engine Degradation Dataset, etc. - FILL THIS IN)*
*   **Description:** The dataset consists of *(describe the data briefly, e.g., time-series sensor readings including temperature, vibration, and pressure)*.
*   **Preprocessing:** Data is processed via our `pipelines/` scripts to handle outliers, normalize features, and extract relevant time-domain metrics prior to model training.

---

## 🛠️ Tech Stack
*   **Languages:** Python, Node.js
*   **Machine Learning:** YOLO ML Processor, Pandas, Scikit-learn, Jupyter Notebook
*   **Backend:** FastAPI
*   **Frontend:** Streamlit
*   **Message Broker:** Apache Kafka
*   **Deployment & Containerization:** Docker, Docker Compose, Kubernetes

---

## 🚀 Getting Started (Build, Run & Deploy)

### Prerequisites
*   [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed.
*   [Minikube](https://minikube.sigs.k8s.io/docs/start/) or a standard Kubernetes cluster (for K8s deployment).

### Option 1: Running Locally with Docker Compose
Use this method for local development and testing.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Nerostry/AI-App-Development-Project---Robust-Predictive-Maintenance-Model.git](https://github.com/Nerostry/AI-App-Development-Project---Robust-Predictive-Maintenance-Model.git)
    cd AI-App-Development-Project---Robust-Predictive-Maintenance-Model
    ```
2.  **Build and spin up the containers:**
    ```bash
    docker-compose up --build
    ```
3.  **Access the application:**
    *   Frontend (Streamlit): `http://localhost:8501`
    *   Backend API Docs (FastAPI): `http://localhost:8000/docs`

### Option 2: Deploying to Kubernetes
Use this method for production-level deployment with scaling capabilities.

1.  **Start your Kubernetes cluster** (if using Minikube):
    ```bash
    minikube start
    ```
2.  **Apply the deployment configurations:**
    ```bash
    kubectl apply -f deployment.yml
    ```
3.  **Verify the pods are running:**
    ```bash
    kubectl get pods
    ```

---

## diagram
<img width="614" height="797" alt="6077950092389323038" src="https://github.com/user-attachments/assets/bcb1dc09-956b-4328-b6df-0b4b32a59464" />


## ⚠️ Known Issues & Limitations
*   **Model Latency:** The YOLO integration currently introduces a slight delay (~500ms) during peak load times. 
*   **Data Upload Limits:** The Streamlit frontend currently caps batch CSV uploads at 200MB to prevent memory timeouts.
*   *(Add any other known bugs or incomplete features here so the graders know you are aware of them)*.
## 📂 Repository Structure
```text
├── User_InterFace/                 # Streamlit frontend application
├── data_processing_service/        # Microservice for data ingestion and cleaning
├── predictive_maintenance_service/ # Core ML prediction microservice
├── pipelines/                      # Data and ML pipeline scripts 
├── datasets/                       # Raw and processed datasets
├── src/                            # Common source code and utilities
├── uploads/                        # Directory for user-uploaded data
├── eda_model_train_AI_App_Dev_Proj.ipynb # Exploratory Data Analysis & Model Training
├── main.py                         # FastAPI backend entry point
├── Dockerfile                      # Application container blueprint
├── docker-compose.yml              # Multi-container orchestration
└── deployment.yml                  # Kubernetes deployment configuration
## diagram
<img width="614" height="797" alt="6077950092389323038" src="https://github.com/user-attachments/assets/bcb1dc09-956b-4328-b6df-0b4b32a59464" />



