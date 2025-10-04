#!/usr/bin/env python3
# setup_models.py - Setup initial models and monitoring

import os
import subprocess
import sys

def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=True)
        else:
            result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e}")
        return False

def main():
    print("🧠 Setting up initial models and monitoring...")
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("[WARNING] Data directory not found. Creating sample data...")
        os.makedirs("data", exist_ok=True)
        
        # Create sample data
        sample_data = """age,gender,bmi,daily_steps,sleep_hours,water_intake_l,calories_consumed,smoker,alcohol,resting_hr,systolic_bp,diastolic_bp,family_history,disease_risk,cholesterol
35,Male,22.5,8000,7,2.0,2200,0,1,72,120,80,0,0,180
45,Female,27.8,5000,6,1.5,1800,1,0,68,135,85,1,1,220
28,Male,24.1,12000,8,2.5,2500,0,0,65,118,75,0,0,170
52,Female,31.2,3000,5,1.0,1600,1,1,75,142,90,1,1,240
31,Male,26.7,7000,7,2.2,2100,0,1,70,128,82,0,0,195"""
        
        with open("data/health_data.csv", "w") as f:
            f.write(sample_data)
        
        print("[INFO] Sample data created at data/health_data.csv")
    
    # Run training
    print("[INFO] Running model training...")
    
    # Build training image
    if not run_command(["docker", "build", "-t", "health-ml-training:latest", "-f", "Dockerfile.training.dockerfile", "."]):
        sys.exit(1)
    
    # Run training container
    cmd = [
        "docker", "run", "-it", "--rm",
        "-v", f"{os.getcwd()}/models:/app/models",
        "-v", f"{os.getcwd()}/mlruns:/app/mlruns", 
        "-v", f"{os.getcwd()}/reports:/app/reports",
        "-v", f"{os.getcwd()}/data:/app/data",
        "health-ml-training:latest"
    ]
    
    if not run_command(cmd):
        sys.exit(1)
    
    print("[INFO] Model training completed!")
    print("[INFO] Models saved to: ./models")
    print("[INFO] MLflow experiments: ./mlruns")
    print("[INFO] Reports: ./reports")
    
    # Restart backend
    print("[INFO] Restarting backend to load new models...")
    if not run_command(["docker-compose", "restart", "ml-backend"]):
        print("[WARNING] Failed to restart backend, but continuing...")
    
    print("✅ Setup completed! Backend is now running with monitoring.")

if __name__ == "__main__":
    main()