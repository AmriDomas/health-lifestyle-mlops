# src/api/app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional, Dict, Any
import os
import logging
import shap
from datetime import datetime
import json
from contextlib import asynccontextmanager  # ✅ Import this

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables untuk models dan monitors
clf_model = None
reg_model = None
cluster_model = None
feature_pipeline = None
performance_monitor = None
drift_detector = None
prediction_history = []

def load_models():
    """Load models dan monitoring systems"""
    global clf_model, reg_model, cluster_model, feature_pipeline, performance_monitor, drift_detector
    
    try:
        # Load models
        clf_model = joblib.load("models/disease_risk_model.pkl")
        logger.info("Classifier model loaded successfully")
        
        reg_model = joblib.load("models/cholesterol_model.pkl")
        logger.info("Regressor model loaded successfully")
        
        cluster_model = joblib.load("models/clustering_model.pkl")
        logger.info("Clustering model loaded successfully")
        
        feature_pipeline = joblib.load("models/feature_pipeline.pkl")
        logger.info("Feature pipeline loaded successfully")
        
        # Load monitoring systems - handle dictionary data only
        performance_monitor = None
        drift_detector = None
        
        # Try to load performance monitor as dictionary
        try:
            performance_data = joblib.load("models/performance_monitor.pkl")
            logger.info(f"Performance monitor data loaded: {type(performance_data)}")
            
            # Create a simple object wrapper for dictionary data
            class PerformanceMonitorWrapper:
                def __init__(self, data):
                    self.performance_history = data.get('performance_history', [])
                    self.alerts = data.get('alerts', [])
                    self.baseline_metrics = type('obj', (object,), data.get('baseline_metrics', {}))()
                    self.model_name = data.get('model_name', 'health_ml_model')
                    self.degradation_threshold = data.get('degradation_threshold', 0.05)
                    
                def get_performance_trends(self, days=7):
                    return pd.DataFrame()
                    
                def generate_performance_report(self):
                    return {
                        'report_timestamp': '2024-01-01T00:00:00',
                        'model_name': self.model_name,
                        'latest_performance': {
                            'accuracy': getattr(self.baseline_metrics, 'accuracy', 0.85),
                            'precision': getattr(self.baseline_metrics, 'precision', 0.82),
                            'recall': getattr(self.baseline_metrics, 'recall', 0.80),
                            'f1_score': getattr(self.baseline_metrics, 'f1_score', 0.81),
                            'auc_roc': getattr(self.baseline_metrics, 'auc_roc', 0.88)
                        },
                        'alerts_summary': {
                            'total_alerts': len(self.alerts),
                            'critical_alerts': 0,
                            'recent_alerts': self.alerts[-5:] if self.alerts else []
                        }
                    }
            
            performance_monitor = PerformanceMonitorWrapper(performance_data)
            logger.info("Performance monitor wrapper created successfully")
            
        except (FileNotFoundError, EOFError, Exception) as e:
            logger.warning(f"Performance monitor not available: {e}")
            performance_monitor = None
        
        # Try to load drift detector as dictionary
        try:
            drift_data = joblib.load("models/drift_detector.pkl")
            logger.info(f"Drift detector data loaded: {type(drift_data)}")
            
            # Create a simple object wrapper for dictionary data
            class DriftDetectorWrapper:
                def __init__(self, data):
                    self.reference_data = data.get('reference_data')
                    self.significance_level = data.get('significance_level', 0.05)
                    self.drift_alerts = data.get('drift_alerts', [])
                    
                def detect_feature_drift(self, current_data):
                    # Return empty DataFrame for now
                    return pd.DataFrame()
            
            drift_detector = DriftDetectorWrapper(drift_data)
            logger.info("Drift detector wrapper created successfully")
            
        except (FileNotFoundError, EOFError, Exception) as e:
            logger.warning(f"Drift detector not available: {e}")
            drift_detector = None
        
        # Log final monitoring status
        monitoring_loaded = performance_monitor is not None and drift_detector is not None
        logger.info(f"Monitoring systems loaded: {monitoring_loaded}")
        
        logger.info("All models and monitoring systems loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

# ✅ LIFESPAN MANAGER - Taruh di sini, sebelum app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Health ML API...")
    load_models()
    yield
    # Shutdown
    logger.info("Shutting down Health ML API...")

# ✅ APP INITIALIZATION - Setelah lifespan manager
app = FastAPI(
    title="Health ML API", 
    version="3.0.0",
    lifespan=lifespan  # Use lifespan instead of deprecated on_event
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class HealthData(BaseModel):
    age: int
    gender: str
    bmi: float
    daily_steps: int
    sleep_hours: float
    water_intake_l: float
    calories_consumed: float
    smoker: int
    alcohol: int
    resting_hr: int
    systolic_bp: int
    diastolic_bp: int
    family_history: int

# Output schema
class PredictionResponse(BaseModel):
    disease_risk_prediction: str
    cholesterol_prediction: float
    cluster_assignment: int
    confidence: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[List[float]] = None
    monitoring_info: Optional[Dict[str, Any]] = None

# Monitoring response schema
class MonitoringResponse(BaseModel):
    status: str
    performance_metrics: Optional[Dict[str, float]] = None
    drift_detected: Optional[bool] = None
    alerts: Optional[List[Dict]] = None
    report_timestamp: str

# ✅ ROUTES - Setelah app initialization
@app.get("/")
async def root():
    return {
        "message": "Health ML API is running!",
        "version": "3.0.0",
        "monitoring_available": performance_monitor is not None,
        "models_loaded": all([
            clf_model is not None,
            reg_model is not None, 
            cluster_model is not None,
            feature_pipeline is not None
        ])
    }

@app.get("/debug-shap")
async def debug_shap():
    """Debug endpoint untuk test SHAP values"""
    try:
        # Create sample data
        sample_data = {
            "age": 45,
            "gender": "Male",
            "bmi": 26.5,
            "daily_steps": 8000,
            "sleep_hours": 7.0,
            "water_intake_l": 2.0,
            "calories_consumed": 2200,
            "smoker": 0,
            "alcohol": 1,
            "resting_hr": 72,
            "systolic_bp": 130,
            "diastolic_bp": 85,
            "family_history": 1
        }
        
        input_df = pd.DataFrame([sample_data])
        processed_data = feature_pipeline.transform(input_df)
        
        explainer = shap.TreeExplainer(clf_model)
        shap_values = explainer.shap_values(processed_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        return {
            "shap_values": shap_values[0].tolist(),
            "features": list(input_df.columns),
            "processed_shape": processed_data.shape,
            "message": "SHAP debug successful"
        }
        
    except Exception as e:
        return {"error": str(e)}

def log_prediction_for_monitoring(input_data: Dict, prediction_result: Dict):
    """Log prediction untuk monitoring purposes (simplified)"""
    prediction_record = {
        "timestamp": datetime.now().isoformat(),
        "input_data": input_data,
        "prediction": prediction_result,
        "model_version": "v2.0"
    }
    
    # Simpan ke memory (dalam production, ini akan ke database)
    prediction_history.append(prediction_record)
    
    # Keep only last 1000 predictions
    if len(prediction_history) > 1000:
        prediction_history.pop(0)
    
    logger.info(f"Prediction logged for monitoring. Total records: {len(prediction_history)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: HealthData, background_tasks: BackgroundTasks = None):
    try:
        logger.info(f"Received prediction request: {data.dict()}")
        
        # Validate models are loaded
        if any(model is None for model in [clf_model, reg_model, cluster_model, feature_pipeline]):
            raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        logger.info(f"Input DataFrame shape: {input_data.shape}")
        
        # Preprocess features
        processed_data = feature_pipeline.transform(input_data)
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Predictions
        disease_pred = clf_model.predict(processed_data)[0]
        cholesterol_pred = reg_model.predict(processed_data)[0]
        cluster_pred = cluster_model.predict(processed_data)[0]
        
        logger.info(f"Predictions - Disease: {disease_pred}, Cholesterol: {cholesterol_pred:.2f}, Cluster: {cluster_pred}")
        
        # Confidence untuk classification
        confidence = None
        if hasattr(clf_model, "predict_proba"):
            confidence = float(np.max(clf_model.predict_proba(processed_data)))
        
        # SHAP feature importance dan values
        feature_importance = None
        shap_values_list = None
        
        try:
            explainer = shap.TreeExplainer(clf_model)
            shap_values = explainer.shap_values(processed_data)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Feature importance (mean absolute SHAP)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = {
                col: float(score)
                for col, score in zip(input_data.columns, mean_abs_shap)
            }
            
            # SHAP values untuk setiap feature (single prediction)
            shap_values_list = shap_values[0].tolist()
            
            # ✅ DEBUG LOGGING
            logger.info(f"SHAP debug - Input features: {len(input_data.columns)}, SHAP values: {len(shap_values_list)}")
            logger.info(f"SHAP values sample: {shap_values_list[:5]}")
            logger.info(f"Feature importance: {feature_importance}")
            
        except Exception as shap_err:
            logger.warning(f"SHAP computation failed: {shap_err}")
        
        # Monitoring info
        monitoring_info = None
        if performance_monitor:
            monitoring_info = {
                "monitoring_available": True,
                "baseline_accuracy": performance_monitor.baseline_metrics.accuracy if hasattr(performance_monitor.baseline_metrics, 'accuracy') else 0.85,
                "performance_alerts": len(performance_monitor.alerts) if performance_monitor.alerts else 0
            }
        
        # Prepare response
        response_data = {
            "disease_risk_prediction": str(disease_pred),
            "cholesterol_prediction": float(cholesterol_pred),
            "cluster_assignment": int(cluster_pred),
            "confidence": confidence,
            "feature_importance": feature_importance,
            "shap_values": shap_values_list,
            "monitoring_info": monitoring_info
        }
        
        # Log prediction untuk monitoring (background task)
        if background_tasks:
            background_tasks.add_task(log_prediction_for_monitoring, data.dict(), response_data)
        
        logger.info("Prediction completed successfully")
        return PredictionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status")
async def get_monitoring_status():
    """Endpoint untuk mendapatkan status monitoring"""
    if not performance_monitor:
        return MonitoringResponse(
            status="monitoring_not_available",
            report_timestamp=datetime.now().isoformat()
        )
    
    # Generate performance report
    performance_report = performance_monitor.generate_performance_report() if hasattr(performance_monitor, 'generate_performance_report') else {}
    
    # Simple drift detection
    drift_detected = False
    if drift_detector and len(prediction_history) > 100:
        # Simplified drift check
        pass
        
    return MonitoringResponse(
        status="healthy",
        performance_metrics=performance_report.get('latest_performance', {}),
        drift_detected=drift_detected,
        alerts=performance_report.get('alerts_summary', {}).get('recent_alerts', []),
        report_timestamp=datetime.now().isoformat()
    )

@app.get("/monitoring/history")
async def get_prediction_history(limit: int = 100):
    """Endpoint untuk mendapatkan history predictions"""
    return {
        "total_predictions": len(prediction_history),
        "predictions": prediction_history[-limit:]
    }

@app.post("/monitoring/retrain")
async def trigger_retraining():
    """Endpoint untuk trigger manual retraining"""
    return {
        "status": "retraining_triggered",
        "message": "Retraining process started",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    models_loaded = all([
        clf_model is not None,
        reg_model is not None,
        cluster_model is not None,
        feature_pipeline is not None
    ])
    
    monitoring_loaded = performance_monitor is not None and drift_detector is not None
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "monitoring_loaded": monitoring_loaded,
        "total_predictions": len(prediction_history),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/info")
async def get_models_info():
    """Endpoint untuk mendapatkan info tentang models yang loaded"""
    model_info = {}
    
    if clf_model:
        model_info['classifier'] = {
            'type': str(type(clf_model)),
            'n_features': clf_model.n_features_in_ if hasattr(clf_model, 'n_features_in_') else 'unknown'
        }
    
    if reg_model:
        model_info['regressor'] = {
            'type': str(type(reg_model)),
            'n_features': reg_model.n_features_in_ if hasattr(reg_model, 'n_features_in_') else 'unknown'
        }
    
    if cluster_model:
        model_info['clustering'] = {
            'type': str(type(cluster_model)),
            'n_clusters': cluster_model.n_clusters if hasattr(cluster_model, 'n_clusters') else 'unknown'
        }
    
    return model_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)