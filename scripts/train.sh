#!/bin/bash

# train.sh - Run model training in Docker

set -e

echo "🧠 Running Model Training in Docker..."

docker build -t health-ml-training:latest -f Dockerfile.training .

# Run training with volume mounts for persistence
docker run -it --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/mlruns:/app/mlruns \
    -v $(pwd)/reports:/app/reports \
    -v $(pwd)/data:/app/data \
    health-ml-training:latest

echo "✅ Training completed! Models saved to ./models"