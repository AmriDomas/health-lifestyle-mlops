@echo off
echo 🚀 Starting MLOPs Quick Demo...
echo.

echo === Starting Services ===
docker-compose up -d
timeout /t 5 /nobreak > nul

echo === Health Check ===
curl http://localhost:8000/health
echo.

echo === Generating Demo Predictions ===
for /L %%i in (1,1,5) do (
    echo Prediction %%i...
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@test_data.json"
    echo.
    timeout /t 1 /nobreak > nul
)

echo === Final Metrics ===
curl http://localhost:9090/api/v1/query?query=ml_predictions_total
echo.

echo 🎉 Demo Ready! Open:
echo - Streamlit: http://localhost:8501
echo - Grafana: http://localhost:3000
echo - Prometheus: http://localhost:9090
echo - MLflow: http://localhost:5000

pause