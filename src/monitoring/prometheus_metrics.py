# src/monitoring/prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, REGISTRY
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    def __init__(self):
        # Prediction metrics - untuk semua model types
        self.predictions_total = Counter(
            'ml_predictions_total',
            'Total number of predictions',
            ['model_type', 'status']
        )
        
        self.prediction_duration = Histogram(
            'ml_prediction_duration_seconds',
            'Prediction duration in seconds',
            ['model_type']
        )
        
        self.prediction_confidence = Histogram(
            'ml_prediction_confidence',
            'Prediction confidence scores',
            ['model_type']
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Model accuracy',
            ['model_type']
        )
        
        self.model_f1_score = Gauge(
            'ml_model_f1_score', 
            'Model F1 score',
            ['model_type']
        )
        
        # Regression specific metrics
        self.regression_mse = Gauge(
            'ml_regression_mse',
            'Regression Mean Squared Error',
            ['model_type']
        )
        
        self.regression_mae = Gauge(
            'ml_regression_mae',
            'Regression Mean Absolute Error', 
            ['model_type']
        )
        
        # Clustering specific metrics
        self.clustering_silhouette = Gauge(
            'ml_clustering_silhouette',
            'Clustering Silhouette Score',
            ['model_type']
        )
        
        # System metrics
        self.active_predictions = Gauge(
            'ml_active_predictions',
            'Number of active predictions'
        )
        
        self.prediction_errors = Counter(
            'ml_prediction_errors_total',
            'Total prediction errors',
            ['model_type', 'error_type']
        )
        
        # Feature metrics
        self.feature_impact = Gauge(
            'ml_feature_impact',
            'Feature impact on predictions',
            ['feature_name', 'model_type']
        )
        
        logger.info("Prometheus metrics initialized")

    def record_prediction(self, model_type: str, duration: float, confidence: float = None):
        """Record prediction metrics untuk semua models"""
        self.predictions_total.labels(model_type=model_type, status='success').inc()
        self.prediction_duration.labels(model_type=model_type).observe(duration)
        
        if confidence is not None:
            self.prediction_confidence.labels(model_type=model_type).observe(confidence)

    def record_prediction_error(self, model_type: str, error_type: str):
        """Record prediction errors untuk semua models"""
        self.predictions_total.labels(model_type=model_type, status='error').inc()
        self.prediction_errors.labels(model_type=model_type, error_type=error_type).inc()

    def update_model_performance(self, model_type: str, accuracy: float = None, f1_score: float = None, 
                               mse: float = None, mae: float = None, silhouette: float = None):
        """Update model performance metrics untuk semua models"""
        if accuracy is not None:
            self.model_accuracy.labels(model_type=model_type).set(accuracy)
        if f1_score is not None:
            self.model_f1_score.labels(model_type=model_type).set(f1_score)
        if mse is not None:
            self.regression_mse.labels(model_type=model_type).set(mse)
        if mae is not None:
            self.regression_mae.labels(model_type=model_type).set(mae)
        if silhouette is not None:
            self.clustering_silhouette.labels(model_type=model_type).set(silhouette)

    def update_feature_impact(self, feature_importance: dict, model_type: str = "disease_risk"):
        """Update feature importance metrics"""
        for feature, importance in feature_importance.items():
            self.feature_impact.labels(feature_name=feature, model_type=model_type).set(importance)

# Global instance
prometheus_metrics = PrometheusMetrics()