# src/retraining/auto_retrain.py
import pandas as pd
import numpy as np
import joblib
import mlflow
from datetime import datetime, timedelta
import logging
from src.models.train_model import ModelTrainer
from src.monitoring.drift_detector import DataDriftDetector

logger = logging.getLogger(__name__)

class AutoRetrainer:
    def __init__(self, drift_threshold: float = 0.3, retrain_interval_days: int = 30):
        self.drift_threshold = drift_threshold
        self.retrain_interval_days = retrain_interval_days
        self.last_retrain_date = None
        
    def should_retrain(self, current_drift_score: float) -> bool:
        """Determine if retraining is needed"""
        # Check drift threshold
        if current_drift_score > self.drift_threshold:
            logger.info(f"Drift threshold exceeded: {current_drift_score}")
            return True
            
        # Check time-based retraining
        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain >= self.retrain_interval_days:
                logger.info(f"Scheduled retraining after {days_since_retrain} days")
                return True
                
        return False
    
    def retrain_pipeline(self, new_data: pd.DataFrame):
        """Execute retraining pipeline"""
        logger.info("Starting auto-retraining pipeline...")
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Retrain models
                trainer = ModelTrainer(experiment_name="health_ml_auto_retrain")
                
                X = new_data.drop(["disease_risk", "cholesterol"], axis=1)
                y_clf = new_data["disease_risk"]
                y_reg = new_data["cholesterol"]
                
                # Retrain models
                clf_model, clf_metrics = trainer.train_classification(X, y_clf)
                reg_model, reg_mse = trainer.train_regression(X, y_reg)
                cluster_model, silhouette, clusters = trainer.train_clustering(X)
                
                # Save new models
                joblib.dump(clf_model, "models/disease_risk_model.pkl")
                joblib.dump(reg_model, "models/cholesterol_model.pkl")
                joblib.dump(cluster_model, "models/clustering_model.pkl")
                
                # Update retrain date
                self.last_retrain_date = datetime.now()
                
                # Log retraining metrics
                mlflow.log_metric("retraining_timestamp", datetime.now().timestamp())
                mlflow.log_metric("dataset_size", len(new_data))
                
                logger.info("Auto-retraining completed successfully")
                
                return True
                
        except Exception as e:
            logger.error(f"Auto-retraining failed: {e}")
            return False

def schedule_retraining():
    """Schedule retraining (to be called periodically)"""
    # Load current data (simulate new data arrival)
    from src.data.data_ingestion import DataIngestion
    ingestion = DataIngestion()
    new_data = ingestion.load_data()
    
    # Load drift detector
    try:
        drift_detector = joblib.load("models/drift_detector.pkl")
        
        # Simulate current data (in practice, this would be new incoming data)
        current_data = new_data.sample(frac=0.3)  # Simulate recent data
        
        # Detect drift
        drift_results = drift_detector.detect_feature_drift(current_data)
        drift_score = drift_results['drift_detected'].mean()
        
        # Check if retraining is needed
        retrainer = AutoRetrainer()
        if retrainer.should_retrain(drift_score):
            logger.info("Initiating auto-retraining...")
            retrainer.retrain_pipeline(new_data)
        else:
            logger.info("No retraining needed at this time")
            
    except FileNotFoundError:
        logger.warning("Drift detector not found. Skipping retraining check.")