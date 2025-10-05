@echo off
echo 🚀 Starting Health ML Pipeline with Advanced Monitoring...

echo [1/5] Starting Backend Services...
docker-compose up -d ml-backend mlflow-ui

echo [2/5] Starting Monitoring Stack...
docker-compose up -d prometheus grafana

echo [3/5] Waiting for services to start...
timeout /t 15 /nobreak

echo [4/5] Testing services...
curl http://localhost:8000/health
curl http://localhost:8000/metrics

echo [5/5] Demo Ready! 🎉
echo.
echo 🌐 Access URLs:
echo    Dashboard:     http://localhost:8501
echo    API Docs:      http://localhost:8000/docs
echo    MLflow:        http://localhost:5000
echo    Prometheus:    http://localhost:9090
echo    Grafana:       http://localhost:3000 (admin/admin)
echo.
echo 📊 Run: streamlit run streamlit_app.py