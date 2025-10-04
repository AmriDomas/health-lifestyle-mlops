@echo off
echo 🧠 Setting up initial models and monitoring...

REM Check if data directory exists
if not exist "data" (
    echo [WARNING] Data directory not found. Creating sample data...
    mkdir data
    echo age,gender,bmi,daily_steps,sleep_hours,water_intake_l,calories_consumed,smoker,alcohol,resting_hr,systolic_bp,diastolic_bp,family_history,disease_risk,cholesterol > data\health_data.csv
    echo 35,Male,22.5,8000,7,2.0,2200,0,1,72,120,80,0,0,180 >> data\health_data.csv
    echo 45,Female,27.8,5000,6,1.5,1800,1,0,68,135,85,1,1,220 >> data\health_data.csv
    echo 28,Male,24.1,12000,8,2.5,2500,0,0,65,118,75,0,0,170 >> data\health_data.csv
    echo 52,Female,31.2,3000,5,1.0,1600,1,1,75,142,90,1,1,240 >> data\health_data.csv
    echo 31,Male,26.7,7000,7,2.2,2100,0,1,70,128,82,0,0,195 >> data\health_data.csv
    echo [INFO] Sample data created at data\health_data.csv
)

echo [INFO] Running model training...
docker build -t health-ml-training:latest -f Dockerfile.training .

docker run -it --rm ^
    -v %CD%\models:/app/models ^
    -v %CD%\mlruns:/app/mlruns ^
    -v %CD%\reports:/app/reports ^
    -v %CD%\data:/app/data ^
    health-ml-training:latest

echo [INFO] Model training completed!
echo [INFO] Models saved to: .\models
echo [INFO] MLflow experiments: .\mlruns  
echo [INFO] Reports: .\reports

echo [INFO] Restarting backend to load new models...
docker-compose restart ml-backend

echo ✅ Setup completed! Backend is now running with monitoring.