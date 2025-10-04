# Tambahkan di app.py untuk real-time monitoring
from src.monitoring import ModelPerformanceMonitor

# Load performance monitor
try:
    performance_monitor = ModelPerformanceMonitor.load_monitor("models/performance_monitor.pkl")
    logger.info("Performance monitor loaded successfully")
except:
    logger.warning("Performance monitor not available")
    performance_monitor = None

# Di endpoint predict, tambahkan monitoring
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: HealthData):
    try:
        # ... existing code ...
        
        # Simpan data untuk monitoring (dalam production, ini akan ke database)
        if performance_monitor:
            # Simpan prediksi untuk monitoring batch
            pass
            
        return PredictionResponse(...)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))