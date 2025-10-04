# Tambahkan di train_model.py setelah training
from src.monitoring import create_performance_monitor

# ... setelah training models ...

# Setup performance monitoring
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

performance_monitor = create_performance_monitor(
    clf_model, 
    feature_pipeline, 
    X_test, 
    y_test,
    model_version="v2.0"
)

# Save monitor untuk digunakan nanti
joblib.dump(performance_monitor, "models/performance_monitor.pkl")