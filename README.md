# 🏥 Health & Lifestyle ML Pipeline

![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)

**One-command deployment:**
```bash
docker-compose up -d
```

End-to-End MLOps pipeline for health risk prediction and lifestyle analysis with real-time monitoring and explainable AI.

## 🚀 Live Demo

[![Demo Video](https://img.shields.io/badge/Watch-Demo-red)](https://linkedin.com/your-post)

## 🏗️ Architecture
🎛️ Streamlit UI → 🚀 FastAPI → 🤖 ML Models → 📊 Monitoring Stack
                                      ↓
              📈 MLflow UI + 📊 Prometheus + 🎯 Grafana


## 🛠️ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/AmriDomas/health-lifestyle-mlops.git
cd health-lifestyle-mlops

# 2. Start backend services
docker-compose up -d ml-backend mlflow-ui

# 3. Start dashboard (new terminal)
streamlit run streamlit_app.py
```

🌐 Access URLs

|Service	                  |URL	                        |Purpose
|--------------------------|-----------------------------|----------------------------------|
|🎛️ Streamlit Dashboard	   |http://localhost:8501	      |Interactive predictions & insights
|🔧 API Documentation	   |http://localhost:8000/docs	|Backend API testing
|📈 MLflow UI	            |http://localhost:5000	      |Experiment tracking
|📊 Prometheus	            |http://localhost:9090	      |Metrics collection
|🎯 Grafana	               |http://localhost:3000	      |Live monitoring dashboards

Default Grafana Login: admin / admin

## 📊 Features

🎯 Multi-Model Predictions
   - Disease Risk Classification - Health risk assessment
   - Cholesterol Regression - Level estimation
   - Lifestyle Clustering - Pattern segmentation

🔍 Explainable AI
    - SHAP Values - Feature importance visualization
    Real-time Explanations - Per-prediction insights
    Impact Analysis - Risk factor identification

📈 Real-time Monitoring
    - Live Metrics - Prediction counts, latency, confidence
    - Performance Tracking - Model accuracy & drift detection
    - Alerting - Automated anomaly detection

🐳 Production Infrastructure
    - Microservices Architecture - Scalable containerized services
    Docker Compose - One-command deployment
    Health Checks - Automated service monitoring

## 🎯 Model Architecture

| Model          | Type                  | Purpose                          |
|----------------|-----------------------|----------------------------------|
| Disease Risk   | Binary Classification | Predict health risk              |
| Cholesterol    | Regression            | Estimate cholesterol levels      |
| Lifestyle      | Clustering            | Segment lifestyle patterns       |

## 🔧 API Endpoints

```http
   POST /predict
   GET /health
   GET /monitoring/status
   GET /models/info
```
## Example Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "bmi": 25.5,
    "daily_steps": 8000,
    "sleep_hours": 7.5,
    "water_intake_l": 2.5,
    "calories_consumed": 2200,
    "smoker": 0,
    "alcohol": 1,
    "resting_hr": 65,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "family_history": 0
  }'
```


## 📁 Project Structure
```text
health-lifestyle-mlops/
├── 🐳 docker-compose.yml          # Full stack deployment
├── 🎛️ streamlit_app.py           # Main dashboard
├── 📊 monitoring_dashboard.py     # Monitoring UI
├── 📁 src/
│   ├── 🚀 api/app.py              # FastAPI backend
│   └── 📈 monitoring/prometheus_metrics.py # Metrics collection
├── 📁 monitoring/
│   ├── prometheus.yaml            # Metrics configuration
│   └── grafana/                   # Dashboard setups
├── 📁 models/                     # ML models (.gitignore)
├── demo_commands.txt              # Video recording guide
├── test_data.json                 # Sample prediction data
├── quick_test.bat                 # Windows demo script
└── live_monitor.ps1               # Real-time monitoring
```
## 🐳 Docker Services

|Service	   |Port	   |Description
|-----------|--------|-------------------------|
|ml-backend	|8000	   |FastAPI ML application   |
|ml-frontend|8501	   |Streamlit dashboard      |
|mlflow-ui	|5000	   |Experiment tracking      |
|prometheus	|9090	   |Metrics database         |
|grafana	   |3000	   |Monitoring dashboards    |


## 🎬 Demo Scripts
Quick Demo (Windows)
```bash
.\quick_test.bat
```
Live Monitoring
```powershell
.\live_monitor.ps1
```
## Step-by-Step Demo
Follow demo_commands.txt for video recording sequence.

## 🛠️ Technology Stack
   - Backend: FastAPI, Python 3.10+
   - Frontend: Streamlit, Plotly, SHAP
   - ML: Scikit-learn, XGBoost, MLflow
   - Monitoring: Prometheus, Grafana
   - Infrastructure: Docker, Docker Compose
   - Testing: Pytest, Requests


## 🤝 Contributing
   - Contributions welcome! Please feel free to submit a Pull Request.
   - Fork the repository
   - Create your feature branch (git checkout -b feature/AmazingFeature)
   - Commit your changes (git commit -m 'Add some AmazingFeature')
   - Push to the branch (git push origin feature/AmazingFeature)
   - Open a Pull Request

## 👨‍💻 Author

Muh Amri Sidiq
AI Engineer | Data Scientist | MLOps Enthusiast

🔗 [Linkedin](http://linkedin.com/in/muh-amri-sidiq)
🔗 [Kaggle](https://www.kaggle.com/amri11)
