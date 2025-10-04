# scripts/create_compatible_monitors.py
import joblib
import pandas as pd
import numpy as np
import os

def create_compatible_monitors():
    """Create monitoring files that are compatible with the backend"""
    print("🔧 Creating compatible monitoring files...")
    
    try:
        os.makedirs('models', exist_ok=True)
        
        # Create performance monitor data structure yang compatible
        # Gunakan dictionary sederhana yang bisa di-load oleh backend
        performance_monitor_data = {
            'performance_history': [],
            'alerts': [],
            'baseline_metrics': {
                'accuracy': 0.85,
                'precision': 0.82, 
                'recall': 0.80,
                'f1_score': 0.81,
                'auc_roc': 0.88,
                'mse': 325.0,
                'mae': 14.2
            },
            'model_name': 'health_ml_model',
            'degradation_threshold': 0.05
        }
        
        joblib.dump(performance_monitor_data, 'models/performance_monitor.pkl')
        print("✅ Performance monitor data created")
        
        # Create drift detector data structure
        drift_detector_data = {
            'reference_data': None,
            'significance_level': 0.05,
            'drift_alerts': []
        }
        
        joblib.dump(drift_detector_data, 'models/drift_detector.pkl')
        print("✅ Drift detector data created")
        
        print("🎉 Compatible monitoring files created!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    create_compatible_monitors()