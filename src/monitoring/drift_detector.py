# src/monitoring/drift_detector.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
import warnings
import joblib
import mlflow
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, significance_level: 0.05):
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.drift_alerts = []
        
    def detect_drift_ks_test(self, current_data: pd.DataFrame, feature: str) -> dict:
        """Detect drift using Kolmogorov-Smirnov test"""
        ref_feature = self.reference_data[feature]
        curr_feature = current_data[feature]
        
        statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
        
        return {
            'feature': feature,
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < self.significance_level,
            'test': 'ks_test'
        }
    
    def detect_drift_psi(self, current_data: pd.DataFrame, feature: str, bins: int = 10) -> dict:
        """Detect drift using Population Stability Index"""
        ref_feature = self.reference_data[feature]
        curr_feature = current_data[feature]
        
        # Create bins based on reference data
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(ref_feature, percentiles)
        
        ref_counts, _ = np.histogram(ref_feature, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_feature, bins=bin_edges)
        
        # Avoid division by zero
        ref_counts = ref_counts + 0.0001
        curr_counts = curr_counts + 0.0001
        
        ref_proportions = ref_counts / len(ref_feature)
        curr_proportions = curr_counts / len(curr_feature)
        
        psi = np.sum((curr_proportions - ref_proportions) * 
                    np.log(curr_proportions / ref_proportions))
        
        return {
            'feature': feature,
            'psi': psi,
            'drift_detected': psi > 0.25,  # Common threshold
            'test': 'psi'
        }
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Detect drift for all features"""
        results = []
        
        for feature in self.reference_data.columns:
            if self.reference_data[feature].dtype in ['float64', 'int64']:
                # Use KS test for numerical features
                result = self.detect_drift_ks_test(current_data, feature)
                results.append(result)
                
        return pd.DataFrame(results)
    
    def log_drift_metrics(self, drift_results: pd.DataFrame, run_id: str = None):
        """Log drift metrics to MLflow"""
        with mlflow.start_run(run_id=run_id, nested=True) as run:
            mlflow.log_param("drift_monitoring_timestamp", datetime.now().isoformat())
            
            for _, row in drift_results.iterrows():
                mlflow.log_metric(f"drift_{row['feature']}_{row['test']}", row['statistic'] if 'statistic' in row else row['psi'])
                mlflow.log_metric(f"drift_detected_{row['feature']}", int(row['drift_detected']))
            
            # Log overall drift score
            drift_score = drift_results['drift_detected'].mean()
            mlflow.log_metric("overall_drift_score", drift_score)
            
            # Alert if significant drift detected
            if drift_score > 0.3:  # 30% of features show drift
                self.drift_alerts.append({
                    'timestamp': datetime.now(),
                    'drift_score': drift_score,
                    'affected_features': drift_results[drift_results['drift_detected']]['feature'].tolist()
                })
                logger.warning(f"Significant data drift detected: {drift_score:.2f}")

class ModelPerformanceMonitor:
    def __init__(self, model, reference_accuracy: float):
        self.model = model
        self.reference_accuracy = reference_accuracy
        self.performance_history = []
        
    def monitor_performance(self, X_test, y_test, prediction_data: dict):
        """Monitor model performance degradation"""
        current_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        
        performance_change = current_accuracy - self.reference_accuracy
        
        alert = None
        if performance_change < -0.05:  # 5% degradation
            alert = {
                'type': 'performance_degradation',
                'severity': 'high' if performance_change < -0.1 else 'medium',
                'current_accuracy': current_accuracy,
                'degradation': performance_change,
                'timestamp': datetime.now()
            }
            logger.warning(f"Model performance degradation: {performance_change:.2f}")
        
        # Log to history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': current_accuracy,
            'degradation': performance_change,
            'alert': alert is not None
        })
        
        return alert

def create_monitoring_dashboard():
    """Create Streamlit dashboard for monitoring"""
    import streamlit as st
    import plotly.express as px
    
    st.title("ML Model Monitoring Dashboard")
    
    # Load monitoring data
    try:
        drift_detector = joblib.load("models/drift_detector.pkl")
        performance_monitor = joblib.load("models/performance_monitor.pkl")
        
        # Drift monitoring
        st.header("📊 Data Drift Monitoring")
        if drift_detector.drift_alerts:
            st.error(f"🚨 Drift Alerts: {len(drift_detector.drift_alerts)}")
            for alert in drift_detector.drift_alerts[-5:]:  # Last 5 alerts
                st.write(f"**{alert['timestamp']}**: Drift score {alert['drift_score']:.2f}")
        else:
            st.success("✅ No significant drift detected")
        
        # Performance monitoring
        st.header("📈 Model Performance")
        if performance_monitor.performance_history:
            history_df = pd.DataFrame(performance_monitor.performance_history)
            fig = px.line(history_df, x='timestamp', y='accuracy', 
                         title='Model Accuracy Over Time')
            st.plotly_chart(fig)
        
    except FileNotFoundError:
        st.warning("Monitoring data not available. Train models first.")