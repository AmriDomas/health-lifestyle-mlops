# src/monitoring/performance_monitor.py
import pandas as pd
import numpy as np
import joblib
import mlflow
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class untuk menyimpan metrics performa"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    mse: float
    mae: float
    predictions: np.ndarray
    actuals: np.ndarray
    model_version: str

class ModelPerformanceMonitor:
    def __init__(self, 
                 model, 
                 feature_pipeline,
                 model_name: str = "health_ml_model",
                 monitoring_window: int = 30,  # days
                 degradation_threshold: float = 0.05,  # 5% degradation
                 sample_size: int = 1000):
        
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.model_name = model_name
        self.monitoring_window = monitoring_window
        self.degradation_threshold = degradation_threshold
        self.sample_size = sample_size
        
        # Storage untuk metrics history
        self.performance_history: List[PerformanceMetrics] = []
        self.alerts: List[Dict] = []
        
        # Reference performance (baseline)
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
    def set_baseline(self, X_test: pd.DataFrame, y_test: pd.DataFrame, model_version: str = "v1.0"):
        """Set baseline performance metrics"""
        logger.info(f"Setting baseline performance for {self.model_name}")
        
        baseline_metrics = self._calculate_metrics(X_test, y_test, model_version)
        self.baseline_metrics = baseline_metrics
        
        # Save baseline to file
        self._save_baseline_metrics(baseline_metrics)
        
        logger.info(f"Baseline set - Accuracy: {baseline_metrics.accuracy:.3f}")
        return baseline_metrics
    
    def _calculate_metrics(self, X: pd.DataFrame, y: pd.DataFrame, model_version: str) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Preprocess features
        X_processed = self.feature_pipeline.transform(X)
        
        # Predictions
        y_pred = self.model.predict(X_processed)
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else 0.0
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            mse=mse,
            mae=mae,
            predictions=y_pred,
            actuals=y.values,
            model_version=model_version
        )
    
    def monitor_performance(self, X_new: pd.DataFrame, y_new: pd.DataFrame, model_version: str = "current") -> Dict:
        """Monitor current performance vs baseline"""
        if self.baseline_metrics is None:
            raise ValueError("Baseline metrics not set. Call set_baseline() first.")
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics(X_new, y_new, model_version)
        
        # Store in history
        self.performance_history.append(current_metrics)
        
        # Compare with baseline
        performance_report = self._compare_with_baseline(current_metrics)
        
        # Check for alerts
        alert = self._check_for_alerts(performance_report, current_metrics)
        if alert:
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert['type']} - {alert['message']}")
        
        # Log to MLflow
        self._log_to_mlflow(current_metrics, performance_report)
        
        return performance_report
    
    def _compare_with_baseline(self, current_metrics: PerformanceMetrics) -> Dict:
        """Compare current metrics with baseline"""
        baseline = self.baseline_metrics
        
        comparisons = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            baseline_val = getattr(baseline, metric_name)
            current_val = getattr(current_metrics, metric_name)
            change = current_val - baseline_val
            change_pct = (change / baseline_val) * 100 if baseline_val != 0 else 0
            
            comparisons[metric_name] = {
                'baseline': baseline_val,
                'current': current_val,
                'change': change,
                'change_pct': change_pct,
                'status': 'improved' if change > 0 else 'degraded' if change < 0 else 'stable'
            }
        
        return {
            'timestamp': current_metrics.timestamp,
            'model_version': current_metrics.model_version,
            'comparisons': comparisons,
            'overall_status': self._calculate_overall_status(comparisons)
        }
    
    def _calculate_overall_status(self, comparisons: Dict) -> str:
        """Calculate overall performance status"""
        degraded_metrics = 0
        total_metrics = len(comparisons)
        
        for metric, data in comparisons.items():
            if data['change_pct'] < -self.degradation_threshold * 100:  # 5% degradation
                degraded_metrics += 1
        
        degradation_ratio = degraded_metrics / total_metrics
        
        if degradation_ratio > 0.5:  # More than 50% metrics degraded
            return 'critical'
        elif degradation_ratio > 0.2:  # More than 20% metrics degraded
            return 'warning'
        else:
            return 'stable'
    
    def _check_for_alerts(self, performance_report: Dict, current_metrics: PerformanceMetrics) -> Optional[Dict]:
        """Check if performance degradation requires alert"""
        if performance_report['overall_status'] in ['warning', 'critical']:
            degraded_metrics = []
            for metric_name, data in performance_report['comparisons'].items():
                if data['change_pct'] < -self.degradation_threshold * 100:
                    degraded_metrics.append(f"{metric_name}: {data['change_pct']:.1f}%")
            
            return {
                'timestamp': datetime.now(),
                'type': 'performance_degradation',
                'severity': performance_report['overall_status'],
                'message': f"Performance degradation detected in {len(degraded_metrics)} metrics",
                'degraded_metrics': degraded_metrics,
                'current_accuracy': current_metrics.accuracy,
                'baseline_accuracy': self.baseline_metrics.accuracy
            }
        
        return None
    
    def _log_to_mlflow(self, current_metrics: PerformanceMetrics, performance_report: Dict):
        """Log performance metrics to MLflow"""
        try:
            with mlflow.start_run(run_name=f"monitoring_{current_metrics.timestamp.strftime('%Y%m%d_%H%M%S')}", nested=True):
                # Log current metrics
                mlflow.log_metric("accuracy", current_metrics.accuracy)
                mlflow.log_metric("precision", current_metrics.precision)
                mlflow.log_metric("recall", current_metrics.recall)
                mlflow.log_metric("f1_score", current_metrics.f1_score)
                mlflow.log_metric("auc_roc", current_metrics.auc_roc)
                mlflow.log_metric("mse", current_metrics.mse)
                mlflow.log_metric("mae", current_metrics.mae)
                
                # Log comparisons
                for metric_name, data in performance_report['comparisons'].items():
                    mlflow.log_metric(f"{metric_name}_change_pct", data['change_pct'])
                
                mlflow.log_param("monitoring_timestamp", current_metrics.timestamp.isoformat())
                mlflow.log_param("overall_status", performance_report['overall_status'])
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def get_performance_trends(self, days: int = 7) -> pd.DataFrame:
        """Get performance trends for the specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_metrics = [
            metrics for metrics in self.performance_history 
            if metrics.timestamp >= cutoff_date
        ]
        
        if not recent_metrics:
            return pd.DataFrame()
        
        trends_data = []
        for metrics in recent_metrics:
            trend_point = {
                'timestamp': metrics.timestamp,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'auc_roc': metrics.auc_roc,
                'model_version': metrics.model_version
            }
            trends_data.append(trend_point)
        
        return pd.DataFrame(trends_data)
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        latest_metrics = self.performance_history[-1]
        trends_df = self.get_performance_trends(30)  # Last 30 days
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'latest_performance': {
                'accuracy': latest_metrics.accuracy,
                'precision': latest_metrics.precision,
                'recall': latest_metrics.recall,
                'f1_score': latest_metrics.f1_score,
                'auc_roc': latest_metrics.auc_roc
            },
            'alerts_summary': {
                'total_alerts': len(self.alerts),
                'critical_alerts': len([a for a in self.alerts if a['severity'] == 'critical']),
                'recent_alerts': [a for a in self.alerts[-5:]]  # Last 5 alerts
            },
            'performance_trends': {
                'accuracy_trend': trends_df['accuracy'].tolist() if not trends_df.empty else [],
                'timestamps': [ts.isoformat() for ts in trends_df['timestamp']] if not trends_df.empty else []
            }
        }
        
        return report
    
    def plot_performance_trends(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot performance trends over time"""
        trends_df = self.get_performance_trends(30)
        
        if trends_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in trends_df.columns:
                axes[i].plot(trends_df['timestamp'], trends_df[metric], marker='o', linewidth=2)
                axes[i].set_title(f'{metric.upper()} Trend')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel(metric.upper())
                axes[i].grid(True, alpha=0.3)
                
                # Add baseline if available
                if self.baseline_metrics:
                    baseline_val = getattr(self.baseline_metrics, metric)
                    axes[i].axhline(y=baseline_val, color='r', linestyle='--', 
                                   label=f'Baseline: {baseline_val:.3f}')
                    axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _save_baseline_metrics(self, baseline_metrics: PerformanceMetrics):
        """Save baseline metrics to file"""
        baseline_data = {
            'timestamp': baseline_metrics.timestamp.isoformat(),
            'accuracy': baseline_metrics.accuracy,
            'precision': baseline_metrics.precision,
            'recall': baseline_metrics.recall,
            'f1_score': baseline_metrics.f1_score,
            'auc_roc': baseline_metrics.auc_roc,
            'mse': baseline_metrics.mse,
            'mae': baseline_metrics.mae,
            'model_version': baseline_metrics.model_version
        }
        
        os.makedirs('models/monitoring', exist_ok=True)
        with open('models/monitoring/baseline_metrics.json', 'w') as f:
            json.dump(baseline_data, f, indent=2)
    
    def load_baseline_metrics(self) -> bool:
        """Load baseline metrics from file"""
        try:
            with open('models/monitoring/baseline_metrics.json', 'r') as f:
                baseline_data = json.load(f)
            
            # Create PerformanceMetrics object from loaded data
            self.baseline_metrics = PerformanceMetrics(
                timestamp=datetime.fromisoformat(baseline_data['timestamp']),
                accuracy=baseline_data['accuracy'],
                precision=baseline_data['precision'],
                recall=baseline_data['recall'],
                f1_score=baseline_data['f1_score'],
                auc_roc=baseline_data['auc_roc'],
                mse=baseline_data['mse'],
                mae=baseline_data['mae'],
                predictions=np.array([]),  # Not saved
                actuals=np.array([]),     # Not saved
                model_version=baseline_data['model_version']
            )
            
            logger.info("Baseline metrics loaded successfully")
            return True
            
        except FileNotFoundError:
            logger.warning("Baseline metrics file not found")
            return False
    
    def save_monitor(self, filepath: str):
        """Save monitor state to file"""
        monitor_state = {
            'model_name': self.model_name,
            'monitoring_window': self.monitoring_window,
            'degradation_threshold': self.degradation_threshold,
            'performance_history_count': len(self.performance_history),
            'alerts_count': len(self.alerts),
            'baseline_set': self.baseline_metrics is not None
        }
        
        joblib.dump(self, filepath)
        logger.info(f"Performance monitor saved to {filepath}")

    @classmethod
    def load_monitor(cls, filepath: str):
        """Load monitor state from file"""
        monitor = joblib.load(filepath)
        logger.info(f"Performance monitor loaded from {filepath}")
        return monitor

# Utility function untuk integrasi mudah
def create_performance_monitor(model, feature_pipeline, X_test, y_test, model_version="v1.0"):
    """Utility function untuk membuat dan setup performance monitor"""
    monitor = ModelPerformanceMonitor(model, feature_pipeline)
    monitor.set_baseline(X_test, y_test, model_version)
    return monitor