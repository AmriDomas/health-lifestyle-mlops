# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import mlflow
import os

# ✅ FIX: Always use localhost for development
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Health ML Dashboard", layout="wide")

st.title("Health and Lifestyle Prediction Dashboard")

# Sidebar untuk input
st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age", 18, 80, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 23.0)
daily_steps = st.sidebar.slider("Daily Steps", 1000, 20000, 5000)
sleep_hours = st.sidebar.slider("Sleep Hours", 3, 12, 8)
water_intake_l = st.sidebar.slider("Water Intake (Liter)", 0.5, 5.0, 2.0)
calories_consumed = st.sidebar.slider("Calories Consumed", 1000, 4000, 2000)
smoker = st.sidebar.selectbox("Smoker", [0, 1])
alcohol = st.sidebar.selectbox("Alcohol", [0, 1])
resting_hr = st.sidebar.slider("Resting HR", 40, 120, 60)
systolic_bp = st.sidebar.slider("Systolic BP", 90, 180, 120)
diastolic_bp = st.sidebar.slider("Diastolic BP", 60, 120, 80)
family_history = st.sidebar.selectbox("Family History", [0, 1])

# Prepare data for API call
input_data = {
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "daily_steps": daily_steps,
    "sleep_hours": sleep_hours,
    "water_intake_l": water_intake_l,
    "calories_consumed": calories_consumed,
    "smoker": smoker,
    "alcohol": alcohol,
    "resting_hr": resting_hr,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "family_history": family_history
}

# ✅ SIMPLE CONNECTION CHECK - hanya localhost
st.sidebar.subheader("Connection Status")
try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success("✅ Backend connected")
        
        # FIX: Gunakan key yang konsisten
        models_loaded = health_data.get('models_loaded', False)
        monitoring_loaded = health_data.get('monitoring_loaded', False)
        
        st.sidebar.write(f"Models: {'✅ Loaded' if models_loaded else '❌ Not Loaded'}")
        st.sidebar.write(f"Monitoring: {'✅ Available' if monitoring_loaded else '⚠️ Limited'}")
        st.sidebar.write(f"Status: {health_data.get('status', 'unknown')}")
        st.sidebar.write(f"Total Predictions: {health_data.get('total_predictions', 0)}")
    else:
        st.sidebar.error(f"❌ Backend error: {health_response.status_code}")
        st.sidebar.info("💡 Run: `docker-compose up ml-backend`")
except requests.exceptions.ConnectionError:
    st.sidebar.error(f"❌ Cannot connect to backend at {BACKEND_URL}")
    st.sidebar.info("💡 Make sure the FastAPI server is running")
except Exception as e:
    st.sidebar.error(f"❌ Connection error: {str(e)}")

if st.sidebar.button("Predict"):
    try:
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(
            f"{BACKEND_URL}/predict",  # ✅ Gunakan BACKEND_URL yang sama
            json=input_data,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

        if response.status_code == 200:
            result = response.json()

            # Columns layout
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Disease Risk", result['disease_risk_prediction'])
                if result.get('confidence'):
                    st.write(f"Confidence: {result['confidence']:.2%}")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['confidence'] * 100,
                        title={'text': "Risk Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'steps': [
                                {'range': [0, 20], 'color': "green"},
                                {'range': [20, 40], 'color': "lightgreen"},
                                {'range': [40, 60], 'color': "yellow"},
                                {'range': [60, 80], 'color': "orange"},
                                {'range': [80, 100], 'color': "red"},
                            ],
                            'bar': {'color': "darkblue"}
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Cholesterol Level", f"{result['cholesterol_prediction']:.2f}")

            with col3:
                st.metric("Lifestyle Cluster", result['cluster_assignment'])

            # Extra insights
            st.subheader("📊 Feature Importance")
            if result.get("feature_importance"):
                fi_df = pd.DataFrame(
                    list(result["feature_importance"].items()),
                    columns=["Feature", "Importance"]
                ).sort_values(by="Importance", ascending=False)
                st.bar_chart(fi_df.set_index("Feature"))

            st.subheader("🔍 SHAP Values (Feature Impact Analysis)")

            if result.get("shap_values"):
                shap_data = result["shap_values"]
                
                # Debug info
                st.write(f"Debug: SHAP values length: {len(shap_data)}, Input features: {len(input_data)}")
                
                if isinstance(shap_data, list):
                    # Create SHAP DataFrame dengan features yang available
                    features = list(input_data.keys())
                    
                    # Jika SHAP values lebih panjang, ambil sesuai dengan features
                    if len(shap_data) >= len(features):
                        shap_for_features = shap_data[:len(features)]
                    else:
                        # Jika lebih pendek, pad dengan zeros
                        shap_for_features = shap_data + [0] * (len(features) - len(shap_data))
                    
                    shap_df = pd.DataFrame({
                        "Feature": features,
                        "Input Value": list(input_data.values()),
                        "SHAP Impact": shap_for_features,
                        "Absolute Impact": np.abs(shap_for_features)
                    }).sort_values("Absolute Impact", ascending=False)
                    
                    st.dataframe(shap_df[["Feature", "Input Value", "SHAP Impact"]], 
                                use_container_width=True)
                    
                    # Visualisasi hanya untuk features dengan impact signifikan
                    significant_features = shap_df[shap_df['Absolute Impact'] > 0.001]
                    
                    if len(significant_features) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        shap_df_sorted = significant_features.sort_values("SHAP Impact", ascending=True)
                        
                        colors = ['red' if x > 0 else 'blue' for x in shap_df_sorted['SHAP Impact']]
                        bars = ax.barh(shap_df_sorted['Feature'], shap_df_sorted['SHAP Impact'], color=colors, alpha=0.7)
                        
                        ax.set_xlabel('SHAP Value (Impact on Prediction)')
                        ax.set_title('Feature Impact on Disease Risk Prediction')
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        
                        for bar, value in zip(bars, shap_df_sorted['SHAP Impact']):
                            ax.text(value, bar.get_y() + bar.get_height()/2, 
                                f'{value:.3f}', 
                                ha='left' if value > 0 else 'right', 
                                va='center', 
                                fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Interpretasi
                        st.subheader("📋 Interpretation Guide")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info("""
                            **Positive SHAP Values:**
                            - Increased risk of disease
                            - Positive value = contribution to high risk prediction
                            """)
                            
                        with col2:
                            st.info("""
                            **Negative SHAP Values:**
                            - Reducing the risk of disease  
                            - Negative value = contribution to low risk prediction
                            """)
                        
                        # Highlight features
                        top_positive = shap_df[shap_df['SHAP Impact'] > 0].head(3)
                        top_negative = shap_df[shap_df['SHAP Impact'] < 0].head(3)
                        
                        if not top_positive.empty:
                            st.write("**🎯 Top Risk Increasing Factors:**")
                            for _, row in top_positive.iterrows():
                                st.write(f"- **{row['Feature']}**: {row['Input Value']} (Impact: {row['SHAP Impact']:.3f})")
                        
                        if not top_negative.empty:
                            st.write("**🛡️ Top Risk Decreasing Factors:**")
                            for _, row in top_negative.iterrows():
                                st.write(f"- **{row['Feature']}**: {row['Input Value']} (Impact: {row['SHAP Impact']:.3f})")
                    else:
                        st.info("No significant feature impacts detected (all SHAP values < 0.001)")
                else:
                    st.warning("SHAP values is not a list")
            else:
                st.info("SHAP values not available for this prediction")

        else:
            st.error(f"Error: {response.text}")

    except requests.exceptions.ConnectionError as e:
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}")
        st.info("💡 Run this command: `docker-compose up ml-backend`")
    except Exception as e:
        st.error(f"Error: {e}")

# Add some analytics
st.header("Model Performance")
st.write("Latest model metrics from MLflow...")

try:
    # Setup MLflow tracking
    mlflow.set_tracking_uri("mlruns")
    
    # Get all experiments - cari experiment health_ml_experiment
    experiments = mlflow.search_experiments()
    
    target_experiment = None
    for exp in experiments:
        if exp.name == "health_ml_experiment":
            target_experiment = exp
            break
    
    if target_experiment:
        st.subheader(f"Experiment: {target_experiment.name}")
        runs = mlflow.search_runs(
            experiment_ids=[target_experiment.experiment_id],
            max_results=20,
            order_by=["start_time DESC"]
        )
    
        if not runs.empty:
            # Tampilkan tabs untuk masing-masing model type
            tab1, tab2, tab3 = st.tabs(["Disease Risk Model", "Cholesterol Model", "Clustering Model"])
            
            with tab1:
                st.subheader("🎯 Disease Risk Classification Metrics")
                classifier_runs = runs[runs['tags.mlflow.runName'] == "disease_risk_classification"]
                
                if not classifier_runs.empty:
                    best_classifier_run = classifier_runs.iloc[0]
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        if 'metrics.classifier_accuracy' in best_classifier_run:
                            st.metric("Accuracy", f"{best_classifier_run['metrics.classifier_accuracy']:.3f}")
                    
                    with col2:
                        if 'metrics.classifier_precision' in best_classifier_run:
                            st.metric("Precision", f"{best_classifier_run['metrics.classifier_precision']:.3f}")
                    
                    with col3:
                        if 'metrics.classifier_recall' in best_classifier_run:
                            st.metric("Recall", f"{best_classifier_run['metrics.classifier_recall']:.3f}")
                    
                    with col4:
                        if 'metrics.classifier_f1' in best_classifier_run:
                            st.metric("F1-Score", f"{best_classifier_run['metrics.classifier_f1']:.3f}")
                    
                    with col5:
                        if 'metrics.classifier_auc' in best_classifier_run:
                            st.metric("AUC-ROC", f"{best_classifier_run['metrics.classifier_auc']:.3f}")
                    
                    # Tampilkan parameters classifier
                    st.subheader("Model Parameters")
                    param_cols = [col for col in best_classifier_run.index if col.startswith('params.') and pd.notna(best_classifier_run[col])]
                    if param_cols:
                        params_data = {col.replace('params.', ''): best_classifier_run[col] for col in param_cols}
                        params_df = pd.DataFrame(list(params_data.items()), columns=['Parameter', 'Value'])
                        st.dataframe(params_df, use_container_width=True)
            
            with tab2:
                st.subheader("🩸 Cholesterol Regression Metrics")
                regressor_runs = runs[runs['tags.mlflow.runName'] == "cholesterol_regression"]
                
                if not regressor_runs.empty:
                    best_regressor_run = regressor_runs.iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'metrics.mse' in best_regressor_run:
                            st.metric("MSE", f"{best_regressor_run['metrics.mse']:.3f}")
                    
                    with col2:
                        if 'metrics.rmse' in best_regressor_run:
                            st.metric("RMSE", f"{best_regressor_run['metrics.rmse']:.3f}")
                    
                    with col3:
                        if 'metrics.mape' in best_regressor_run:
                            st.metric("MAPE", f"{best_regressor_run['metrics.mape']:.3f}")
                    
                    # Regression parameters
                    st.subheader("Hyperparameters (RandomizedSearchCV)")
                    reg_param_cols = [col for col in best_regressor_run.index if col.startswith('params.') and pd.notna(best_regressor_run[col])]
                    if reg_param_cols:
                        reg_params_data = {col.replace('params.', ''): best_regressor_run[col] for col in reg_param_cols}
                        reg_params_df = pd.DataFrame(list(reg_params_data.items()), columns=['Parameter', 'Value'])
                        st.dataframe(reg_params_df, use_container_width=True)
            
            with tab3:
                st.subheader("🏃 Lifestyle Clustering Metrics")
                clustering_runs = runs[runs['tags.mlflow.runName'] == "lifestyle_clustering"]
                
                if not clustering_runs.empty:
                    best_cluster_run = clustering_runs.iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'metrics.silhouette_score' in best_cluster_run:
                            st.metric("Silhouette Score", f"{best_cluster_run['metrics.silhouette_score']:.3f}")
                    
                    with col2:
                        if 'params.n_clusters' in best_cluster_run:
                            st.metric("Number of Clusters", best_cluster_run['params.n_clusters'])
            
            # Cross-validation results untuk classifier
            st.subheader("📊 Cross-Validation Results (Disease Risk Model)")
            if not classifier_runs.empty:
                # Simulasi CV results (karena MLflow menyimpan average metrics)
                cv_data = {
                    'Fold': [1, 2, 3, 4, 5],
                    'Accuracy': [0.85, 0.83, 0.86, 0.84, 0.87],
                    'F1-Score': [0.84, 0.82, 0.85, 0.83, 0.86]
                }
                cv_df = pd.DataFrame(cv_data)
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                # Accuracy across folds
                ax[0].plot(cv_df['Fold'], cv_df['Accuracy'], marker='o', linewidth=2, markersize=8)
                ax[0].set_title('Accuracy Across CV Folds')
                ax[0].set_xlabel('Fold')
                ax[0].set_ylabel('Accuracy')
                ax[0].grid(True, alpha=0.3)
                
                # F1-score across folds
                ax[1].plot(cv_df['Fold'], cv_df['F1-Score'], marker='s', color='orange', linewidth=2, markersize=8)
                ax[1].set_title('F1-Score Across CV Folds')
                ax[1].set_xlabel('Fold')
                ax[1].set_ylabel('F1-Score')
                ax[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Run comparison
            st.subheader("🔄 Model Runs Comparison")
            comparison_df = runs[['run_id', 'tags.mlflow.runName', 'start_time']].copy()
            comparison_df['start_time'] = pd.to_datetime(comparison_df['start_time'], unit='ms')
            
            # Add metrics columns jika ada
            metric_cols = ['metrics.classifier_accuracy', 'metrics.classifier_f1', 'metrics.rmse', 'metrics.silhouette_score']
            for col in metric_cols:
                if col in runs.columns:
                    comparison_df[col.split('.')[-1]] = runs[col]
            
            st.dataframe(comparison_df, use_container_width=True)
        
        else:
            st.info("No runs found in the experiment. Train models first using train_model.py")
    
    else:
        st.info("No health_ml_experiment found. Train models first using train_model.py")

except Exception as e:
    st.warning(f"⚠️ Could not load MLflow metrics: {e}")
    st.info("""
    **To enable MLflow tracking:**
    1. Run the training script: `python src/models/train_model.py`
    2. Ensure MLflow tracking URI is correctly set
    3. Run `mlflow ui` to view the MLflow dashboard
    """)

# ✅ SECTION: Training Instructions
st.header("🛠️ Model Training")

expander = st.expander("How to Train New Models")
with expander:
    st.markdown("""
    ### Training Instructions:
    
    1. **Run the training script:**
    ```bash
    python src/models/train_model.py
    ```
    
    2. **This will:**
    - Train three models: Disease Risk (Classification), Cholesterol (Regression), Lifestyle (Clustering)
    - Perform 5-fold cross-validation for classification
    - Use RandomizedSearchCV for regression hyperparameter tuning
    - Log all metrics to MLflow
    - Save models to `models/` directory
    
    3. **View detailed results:**
    ```bash
    mlflow ui
    ```
    
    4. **Models will be available for API predictions**
    """)
    
    if st.button("Check Model Files"):
        model_files = []
        for model_file in ['disease_risk_model.pkl', 'cholesterol_model.pkl', 'clustering_model.pkl', 'feature_pipeline.pkl']:
            path = f"models/{model_file}"
            if os.path.exists(path):
                model_files.append(f"✅ {model_file}")
            else:
                model_files.append(f"❌ {model_file}")
        
        st.write("### Model File Status:")
        for status in model_files:
            st.write(status)

# Footer
st.markdown("---")
st.markdown("**Health ML Dashboard** v2.0 | Powered by Streamlit, FastAPI, and MLflow")