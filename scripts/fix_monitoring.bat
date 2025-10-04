@echo off
echo 🔧 Fixing Monitoring Setup...

echo [INFO] Checking for monitoring files...
if exist "models\performance_monitor.pkl" (
    echo [INFO] Performance monitor found locally
) else (
    echo [ERROR] Performance monitor not found! Please run training first.
    exit /b 1
)

if exist "models\drift_detector.pkl" (
    echo [INFO] Drift detector found locally
) else (
    echo [ERROR] Drift detector not found! Please run training first.
    exit /b 1
)

echo [INFO] Restarting backend to load monitoring files...
docker-compose restart ml-backend

echo [INFO] Waiting for backend to start...
timeout /t 10 /nobreak >nul

echo [INFO] Testing backend health...
curl http://localhost:8000/health

echo ✅ Monitoring setup completed!