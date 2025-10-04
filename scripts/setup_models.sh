#!/bin/bash

# setup_models.sh - Setup initial models and monitoring

set -e

echo "🧠 Setting up initial models and monitoring..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if data directory exists
if [ ! -d "data" ]; then
    print_warning "Data directory not found. Creating sample data..."
    mkdir -p data
    # Create sample data file (you should replace this with your actual data)
    cat > data/health_data.csv << EOF
age,gender,bmi,daily_steps,sleep_hours,water_intake_l,calories_consumed,smoker,alcohol,resting_hr,systolic_bp,diastolic_bp,family_history,disease_risk,cholesterol
35,Male,22.5,8000,7,2.0,2200,0,1,72,120,80,0,0,180
45,Female,27.8,5000,6,1.5,1800,1,0,68,135,85,1,1,220
28,Male,24.1,12000,8,2.5,2500,0,0,65,118,75,0,0,170
52,Female,31.2,3000,5,1.0,1600,1,1,75,142,90,1,1,240
31,Male,26.7,7000,7,2.2,2100,0,1,70,128,82,0,0,195
EOF
    print_status "Sample data created at data/health_data.csv"
fi

# Run training
print_status "Running model training..."
docker build -t health-ml-training:latest -f Dockerfile.training .

docker run -it --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/mlruns:/app/mlruns \
    -v $(pwd)/reports:/app/reports \
    -v $(pwd)/data:/app/data \
    health-ml-training:latest

print_status "✅ Model training completed!"
print_status "📊 Models saved to: ./models"
print_status "📈 MLflow experiments: ./mlruns"
print_status "📋 Reports: ./reports"

# Restart backend to load new models
print_status "Restarting backend to load new models..."
docker-compose restart ml-backend

print_status "✅ Setup completed! Backend is now running with monitoring."