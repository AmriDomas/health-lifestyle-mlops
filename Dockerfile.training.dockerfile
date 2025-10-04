# Dockerfile.training
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for training
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p models mlruns reports data

# Set environment variables for training
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV GIT_PYTHON_REFRESH=quiet

# Command to run training
CMD ["python", "src/models/train_model.py"]