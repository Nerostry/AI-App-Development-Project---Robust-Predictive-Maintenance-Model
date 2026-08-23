FROM python:3.12-slim

WORKDIR /app

# System dependencies (kept minimal; add build-essential only if a wheel needs compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY predictive_maintenance_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared pipelines package (app.py imports pipelines.model_train)
COPY pipelines/ ./pipelines/

# Copy the service source code
COPY predictive_maintenance_service/src/ ./src/

# Copy trained model weights (created by pipelines/model_train.py)
COPY saved_models/ ./saved_models/

ENV MODEL_DIR=/app/saved_models
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]