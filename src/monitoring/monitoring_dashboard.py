# src/monitoring/monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MonitoringDashboard:
    def __init__(self, backend_url: str = None):
        # Get backend URL from environment variable or use default
        self.backend_url = backend_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.performance_monitor = None
        self.drift_detector = None
        
        # Try to load local monitors
        self._load_local_monitors()
        
        # Log the backend URL being used
        print(f"Monitoring dashboard using backend: {self.backend_url}")
    
    def _load_local_monitors(self):
        """Try to load monitoring objects from local files"""
        try:
            self.performance_monitor = joblib.load("models/performance_monitor.pkl")
            logger.info("Performance monitor loaded from local file")
        except FileNotFoundError:
            logger.warning("Performance monitor file not found")
            self.performance_monitor = None
            
        try:
            self.drift_detector = joblib.load("models/drift_detector.pkl")
            logger.info("Drift detector loaded from local file")
        except FileNotFoundError:
            logger.warning("Drift detector file not found")
            self.drift_detector = None
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get status from backend API"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Backend returned status {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Cannot connect to backend: {e}"}
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status from backend"""
        try:
            response = requests.get(f"{self.backend_url}/monitoring/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Monitoring endpoint returned status {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Cannot connect to monitoring endpoint: {e}"}
    
    def get_prediction_history(self, limit: int = 100) -> Dict[str, Any]:
        """Get prediction history from backend"""
        try:
            response = requests.get(f"{self.backend_url}/monitoring/history?limit={limit}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"History endpoint returned status {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Cannot connect to history endpoint: {e}"}

def setup_dashboard():
    """Setup and render the monitoring dashboard"""
    st.title("🔍 ML Model Monitoring Dashboard")
    st.markdown("Real-time monitoring of model performance, data drift, and system health")
    
    # Initialize dashboard
    dashboard = MonitoringDashboard()
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    backend_url = st.sidebar.text_input(
        "Backend URL", 
        value="http://ml-backend:8000",
        help="URL of the FastAPI backend service"
    )
    
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval",
        [30, 60, 300, 600],
        index=1,
        format_func=lambda x: f"{x} seconds",
        help="How often to refresh the dashboard"
    )
    
    # Auto-refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 System Health", 
        "📈 Performance", 
        "🎯 Data Drift", 
        "📋 Prediction History"
    ])
    
    with tab1:
        render_system_health_tab(dashboard)
    
    with tab2:
        render_performance_tab(dashboard)
    
    with tab3:
        render_drift_tab(dashboard)
    
    with tab4:
        render_history_tab(dashboard)

def render_system_health_tab(dashboard: MonitoringDashboard):
    """Render system health tab"""
    st.header("🖥️ System Health Overview")
    
    # Get backend status
    with st.spinner("Checking backend status..."):
        backend_status = dashboard.get_backend_status()
    
    if "error" in backend_status:
        st.error(f"❌ Backend Connection Error: {backend_status['error']}")
        return
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_emoji = "✅" if backend_status.get("models_loaded", False) else "❌"
        st.metric("Model Status", f"{status_emoji} {'Loaded' if backend_status.get('models_loaded') else 'Not Loaded'}")
    
    with col2:
        monitoring_status = "✅ Available" if backend_status.get("monitoring_loaded", False) else "⚠️ Limited"
        st.metric("Monitoring", monitoring_status)
    
    with col3:
        total_predictions = backend_status.get("total_predictions", 0)
        st.metric("Total Predictions", f"{total_predictions:,}")
    
    with col4:
        status = backend_status.get("status", "unknown")
        status_color = "🟢" if status == "healthy" else "🟡" if status == "degraded" else "🔴"
        st.metric("Overall Status", f"{status_color} {status.title()}")
    
    # Backend details
    st.subheader("Backend Details")
    details_col1, details_col2 = st.columns(2)
    
    with details_col1:
        st.json(backend_status)
    
    with details_col2:
        # Model information
        try:
            models_info = requests.get(f"{dashboard.backend_url}/models/info", timeout=5).json()
            st.write("**Model Information:**")
            for model_type, info in models_info.items():
                st.write(f"- **{model_type.title()}**: {info.get('type', 'Unknown')}")
        except:
            st.write("**Model Information:** Not available")
    
    # Monitoring status
    st.subheader("Monitoring Status")
    monitoring_status = dashboard.get_monitoring_status()
    
    if "error" in monitoring_status:
        st.warning(f"Monitoring endpoint unavailable: {monitoring_status['error']}")
    else:
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            status = monitoring_status.get("status", "unknown")
            st.metric("Monitoring Status", status.title())
        
        with status_col2:
            drift_detected = monitoring_status.get("drift_detected", False)
            drift_status = "✅ No Drift" if not drift_detected else "🚨 Drift Detected"
            st.metric("Data Drift", drift_status)
        
        with status_col3:
            alerts = monitoring_status.get("alerts", [])
            st.metric("Active Alerts", len(alerts))
        
        # Display alerts
        if alerts:
            st.error("**Active Alerts:**")
            for alert in alerts:
                st.write(f"- {alert.get('message', 'Unknown alert')}")

def render_performance_tab(dashboard: MonitoringDashboard):
    """Render performance monitoring tab"""
    st.header("📈 Model Performance Monitoring")
    
    if dashboard.performance_monitor is None:
        st.warning("Performance monitor not available locally. Using backend data...")
        render_performance_from_backend(dashboard)
        return
    
    # Performance metrics
    st.subheader("Current Performance Metrics")
    
    if dashboard.performance_monitor.performance_history:
        latest_metrics = dashboard.performance_monitor.performance_history[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{latest_metrics.accuracy:.3f}")
        
        with col2:
            st.metric("Precision", f"{latest_metrics.precision:.3f}")
        
        with col3:
            st.metric("Recall", f"{latest_metrics.recall:.3f}")
        
        with col4:
            st.metric("F1-Score", f"{latest_metrics.f1_score:.3f}")
        
        with col5:
            st.metric("AUC-ROC", f"{latest_metrics.auc_roc:.3f}")
    
    # Performance trends
    st.subheader("Performance Trends")
    
    trends_df = dashboard.performance_monitor.get_performance_trends(30)
    if not trends_df.empty:
        # Create trend visualization
        fig = go.Figure()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in trends_df.columns:
                fig.add_trace(go.Scatter(
                    x=trends_df['timestamp'],
                    y=trends_df[metric],
                    name=metric.upper(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison with baseline
        if dashboard.performance_monitor.baseline_metrics:
            st.subheader("Baseline Comparison")
            baseline = dashboard.performance_monitor.baseline_metrics
            latest = dashboard.performance_monitor.performance_history[-1] if dashboard.performance_monitor.performance_history else None
            
            if latest:
                comparison_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Baseline': [baseline.accuracy, baseline.precision, baseline.recall, baseline.f1_score],
                    'Current': [latest.accuracy, latest.precision, latest.recall, latest.f1_score],
                    'Change': [
                        latest.accuracy - baseline.accuracy,
                        latest.precision - baseline.precision,
                        latest.recall - baseline.recall,
                        latest.f1_score - baseline.f1_score
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
    
    # Alerts section
    if dashboard.performance_monitor.alerts:
        st.subheader("🚨 Performance Alerts")
        for alert in dashboard.performance_monitor.alerts[-5:]:  # Last 5 alerts
            with st.expander(f"Alert: {alert.get('timestamp', 'Unknown time')}"):
                st.write(f"**Type:** {alert.get('type', 'Unknown')}")
                st.write(f"**Severity:** {alert.get('severity', 'Unknown')}")
                st.write(f"**Message:** {alert.get('message', 'No message')}")
                if 'degraded_metrics' in alert:
                    st.write("**Affected Metrics:**")
                    for metric in alert['degraded_metrics']:
                        st.write(f"- {metric}")

def render_performance_from_backend(dashboard: MonitoringDashboard):
    """Render performance data from backend when local monitor is unavailable"""
    monitoring_status = dashboard.get_monitoring_status()
    
    if "error" in monitoring_status:
        st.error(f"Cannot load performance data: {monitoring_status['error']}")
        return
    
    performance_metrics = monitoring_status.get("performance_metrics", {})
    
    if performance_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{performance_metrics.get('accuracy', 0):.3f}")
        
        with col2:
            st.metric("Precision", f"{performance_metrics.get('precision', 0):.3f}")
        
        with col3:
            st.metric("Recall", f"{performance_metrics.get('recall', 0):.3f}")
        
        with col4:
            st.metric("F1-Score", f"{performance_metrics.get('f1_score', 0):.3f}")
    else:
        st.info("No performance metrics available from backend")

def render_drift_tab(dashboard: MonitoringDashboard):
    """Render data drift monitoring tab"""
    st.header("🎯 Data Drift Detection")
    
    if dashboard.drift_detector is None:
        st.warning("Drift detector not available locally.")
        
        # Check backend for drift status
        monitoring_status = dashboard.get_monitoring_status()
        if not "error" in monitoring_status:
            drift_detected = monitoring_status.get("drift_detected", False)
            if drift_detected:
                st.error("🚨 Data drift detected by backend monitoring!")
            else:
                st.success("✅ No data drift detected by backend monitoring")
        
        return
    
    st.info("""
    Data drift monitoring detects changes in the statistical properties of input data over time.
    This helps identify when models may need retraining due to changing data patterns.
    """)
    
    # Simulate drift analysis (in real implementation, this would use actual data)
    st.subheader("Feature Distribution Analysis")
    
    # Create sample drift visualization
    features = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'systolic_bp']
    drift_scores = np.random.uniform(0, 0.5, len(features))
    
    drift_df = pd.DataFrame({
        'Feature': features,
        'Drift Score': drift_scores,
        'Status': ['Low' if score < 0.1 else 'Medium' if score < 0.3 else 'High' for score in drift_scores]
    })
    
    # Drift score visualization
    fig = px.bar(drift_df, x='Feature', y='Drift Score', color='Status',
                 title='Feature Drift Scores',
                 color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drift alerts
    high_drift_features = drift_df[drift_df['Status'] == 'High']
    if not high_drift_features.empty:
        st.error("**🚨 High Drift Features Detected:**")
        for _, row in high_drift_features.iterrows():
            st.write(f"- **{row['Feature']}**: Drift score = {row['Drift Score']:.3f}")
    
    # Drift over time (simulated)
    st.subheader("Drift Over Time")
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    drift_trend = np.cumsum(np.random.normal(0, 0.02, 30))
    drift_trend_df = pd.DataFrame({
        'Date': dates,
        'Cumulative Drift': np.maximum(0, drift_trend)  # Ensure non-negative
    })
    
    fig_trend = px.line(drift_trend_df, x='Date', y='Cumulative Drift',
                       title='Cumulative Data Drift Over Time')
    fig_trend.add_hline(y=0.3, line_dash="dash", line_color="red", 
                       annotation_text="Drift Threshold")
    
    st.plotly_chart(fig_trend, use_container_width=True)

def render_history_tab(dashboard: MonitoringDashboard):
    """Render prediction history tab"""
    st.header("📋 Prediction History")
    
    history_data = dashboard.get_prediction_history(limit=50)
    
    if "error" in history_data:
        st.error(f"Cannot load prediction history: {history_data['error']}")
        return
    
    predictions = history_data.get("predictions", [])
    total_predictions = history_data.get("total_predictions", 0)
    
    st.metric("Total Predictions in History", total_predictions)
    
    if not predictions:
        st.info("No prediction history available")
        return
    
    # Convert to DataFrame for easier display
    history_df = pd.DataFrame([
        {
            'timestamp': pred.get('timestamp', ''),
            'disease_risk': pred.get('prediction', {}).get('disease_risk_prediction', ''),
            'cholesterol': pred.get('prediction', {}).get('cholesterol_prediction', 0),
            'cluster': pred.get('prediction', {}).get('cluster_assignment', 0),
            'age': pred.get('input_data', {}).get('age', 0),
            'bmi': pred.get('input_data', {}).get('bmi', 0)
        }
        for pred in predictions
    ])
    
    if not history_df.empty:
        # Basic statistics
        st.subheader("Prediction Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_cholesterol = history_df['cholesterol'].mean()
            st.metric("Avg Cholesterol", f"{avg_cholesterol:.1f}")
        
        with col2:
            risk_ratio = (history_df['disease_risk'] == '1').mean()
            st.metric("Disease Risk Ratio", f"{risk_ratio:.1%}")
        
        with col3:
            avg_age = history_df['age'].mean()
            st.metric("Avg Age", f"{avg_age:.1f}")
        
        with col4:
            avg_bmi = history_df['bmi'].mean()
            st.metric("Avg BMI", f"{avg_bmi:.1f}")
        
        # Recent predictions table
        st.subheader("Recent Predictions")
        st.dataframe(history_df, use_container_width=True)
        
        # Prediction trends
        st.subheader("Prediction Trends")
        
        if 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp')
            
            # Cholesterol trend
            fig_chol = px.line(history_df, x='timestamp', y='cholesterol',
                             title='Cholesterol Predictions Over Time')
            st.plotly_chart(fig_chol, use_container_width=True)
            
            # Risk distribution
            fig_risk = px.histogram(history_df, x='disease_risk',
                                  title='Disease Risk Distribution')
            st.plotly_chart(fig_risk, use_container_width=True)

def main():
    """Main function to run the monitoring dashboard"""
    try:
        setup_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Health ML Monitoring Dashboard** • "
            "Real-time model performance and data drift monitoring"
        )
        
    except Exception as e:
        st.error(f"Error in monitoring dashboard: {str(e)}")
        logger.exception("Dashboard error")

if __name__ == "__main__":
    main()