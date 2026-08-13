apiVersion: apps/v1
kind: Deployment
metadata:
  name: predictive-maintenance-app
  labels:
    app: predictive-maintenance
spec:
  replicas: 1
  selector:
    matchLabels:
      app: predictive-maintenance
  template:
    metadata:
      labels:
        app: predictive-maintenance
    spec:
      containers:
      - name: predictive-maintenance-container
        image: your-docker-registry/predictive-maintenance:latest # Replace with your image repository path
        imagePullPolicy: Always
        
        # Environment variables mirrored from Dockerfile
        env:
        - name: PYTHONDONTWRITEBYTECODE
          value: "1"
        - name: PYTHONUNBUFFERED
          value: "1"

        # Resource limits and requests to prevent pod evictions
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"

        # Persistent storage mounts for output/data directories defined in Dockerfile
        volumeMounts:
        - name: app-storage
          mountPath: /app/datasets
          subPath: datasets
        - name: app-storage
          mountPath: /app/ingested_datasets
          subPath: ingested_datasets
        - name: app-storage
          mountPath: /app/uploads
          subPath: uploads
        - name: app-storage
          mountPath: /app/models
          subPath: models

      # PVC storage configuration
      volumes:
      - name: app-storage
        persistentVolumeClaim:
          claimName: predictive-maintenance-pvc