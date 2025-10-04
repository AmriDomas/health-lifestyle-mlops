# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib

st.set_page_config(page_title="ML Performance Monitoring", layout="wide")

st.title("🔍 ML Model Performance Monitoring Dashboard")

try:
    # Load monitors
    performance_monitor = joblib.load("models/performance_monitor.pkl")
    drift_detector = joblib.load("models/drift_detector.pkl")
    
    # Performance Overview
    st.header("📊 Performance Overview")
    
    if performance_monitor.performance_history:
        latest_metrics = performance_monitor.performance_history[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{latest_metrics.accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{latest_metrics.precision:.3f}")
        with col3:
            st.metric("Recall", f"{latest_metrics.recall:.3f}")
        with col4:
            st.metric("F1-Score", f"{latest_metrics.f1_score:.3f}")
    
    # Alerts Section
    st.header("🚨 Alerts & Issues")
    
    if performance_monitor.alerts:
        for alert in performance_monitor.alerts[-3:]:  # Last 3 alerts
            st.error(f"**{alert['timestamp']}** - {alert['message']}")
    else:
        st.success("No critical alerts")
    
    # Performance Trends
    st.header("📈 Performance Trends")
    trends_df = performance_monitor.get_performance_trends(30)
    if not trends_df.empty:
        fig = px.line(trends_df, x='timestamp', y=['accuracy', 'precision', 'recall', 'f1_score'],
                     title='Model Performance Over Time')
        st.plotly_chart(fig)
    
except FileNotFoundError:
    st.warning("Monitoring data not available. Train models and setup monitoring first.")