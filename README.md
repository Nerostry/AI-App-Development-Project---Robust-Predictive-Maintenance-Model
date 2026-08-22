# AI App Development Project: Robust Predictive Maintenance Model

Welcome to the **Robust Predictive Maintenance Model** repository. This project aims to deliver a scalable, end-to-end machine learning application that predicts maintenance needs. It features a robust data pipeline for seamless data ingestion, a trained predictive model, and a containerized microservices architecture for easy deployment.

**Created by the Integrated Legacy 4.0 Team:** 
* Eshimeet Kaur (@eiooioo)
* Gina (@jiinani)
* Jun Hian (@Nerostry)

---

## 🚀 Key Features

* **Predictive Maintenance ML Model:** Custom machine learning models (including YOLO integration) for accurate failure prediction.
* **Robust Data Pipeline:** Automated pipelines for data ingestion, cleaning, and preprocessing.
* **Microservices Architecture:** Independently scalable services communicating via Kafka.
* **Interactive Frontend:** User-friendly interface built with Streamlit.
* **High-Performance Backend:** API endpoints powered by FastAPI.
* **Containerized Deployment:** Fully dockerized environment using Docker Compose and Kubernetes deployment configurations.

---

## 🛠️ Tech Stack

* **Languages:** Python, JavaScript (Node.js for specific services/configs)
* **Machine Learning:** Jupyter Notebook, YOLO ML Processor, Pandas, Scikit-learn
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Message Broker:** Apache Kafka
* **Deployment & Containerization:** Docker, Docker Compose, Kubernetes (`deployment.yml`)

---

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