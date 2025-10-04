# 🏥 Health & Lifestyle ML Pipeline

![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)

End-to-End MLOps pipeline for health risk prediction and lifestyle analysis with real-time monitoring and explainable AI.

## 🚀 Live Demo

[![Demo Video](https://img.shields.io/badge/Watch-Demo-red)](https://linkedin.com/your-post)

## 🏗️ Architecture
User → Streamlit Dashboard → FastAPI Backend → ML Models
↓
MLflow UI + Monitoring


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

Access URLs:

📊 Dashboard: http://localhost:8501

🔧 API Docs: http://localhost:8000/docs

📈 MLflow: http://localhost:5000

## 📊 Features
   - 🎯 Multi-Model Predictions: Disease risk, cholesterol, lifestyle clustering
   - 🔍 Explainable AI: SHAP values for feature importance
   - 📈 Real-time Monitoring: Performance tracking & drift detection
   - 🐳 Full Containerization: Docker & Docker Compose
   - 🔄 Experiment Tracking: MLflow integration
   - ⚡ Production Ready: FastAPI backend with monitoring

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
## 📁 Project Structure
```text
health-ml-pipeline/
├── src/
│   ├── api/           # FastAPI backend
│   ├── models/        # Training pipelines
│   ├── features/      # Feature engineering
│   └── monitoring/    # Performance tracking
├── scripts/           # Utility scripts
├── tests/            # Test suites
└── docker-compose.yml
```
## 🐳 Docker Services
  - ml-backend: FastAPI application (port 8000)
  - mlflow-ui: Experiment tracking (port 5000)
  - ml-frontend: Streamlit dashboard (port 8501)

## 🤝 Contributing
Contributions welcome! Please feel free to submit a Pull Request.

## 👨‍💻 Author

Muh Amri Sidiq
AI Engineer | Data Scientist | MLOps Enthusiast

🔗 [Linkedin](http://linkedin.com/in/muh-amri-sidiq)
🔗 [Kaggle](https://www.kaggle.com/amri11)
