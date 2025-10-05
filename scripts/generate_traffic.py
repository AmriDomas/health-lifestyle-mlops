# scripts/generate_traffic.py
import requests
import time
import random

def generate_traffic():
    base_url = "http://localhost:8000"
    
    # Test data samples
    test_samples = [
        {
            "age": 35, "gender": "Male", "bmi": 22.5, "daily_steps": 8000,
            "sleep_hours": 7.0, "water_intake_l": 2.0, "calories_consumed": 2200,
            "smoker": 0, "alcohol": 1, "resting_hr": 72, "systolic_bp": 120,
            "diastolic_bp": 80, "family_history": 0
        },
        {
            "age": 45, "gender": "Female", "bmi": 28.0, "daily_steps": 5000, 
            "sleep_hours": 6.0, "water_intake_l": 1.5, "calories_consumed": 1800,
            "smoker": 1, "alcohol": 0, "resting_hr": 75, "systolic_bp": 135,
            "diastolic_bp": 85, "family_history": 1
        },
        {
            "age": 28, "gender": "Male", "bmi": 24.0, "daily_steps": 12000,
            "sleep_hours": 8.0, "water_intake_l": 2.5, "calories_consumed": 2500, 
            "smoker": 0, "alcohol": 0, "resting_hr": 65, "systolic_bp": 118,
            "diastolic_bp": 75, "family_history": 0
        }
    ]
    
    print("🚀 Generating traffic for monitoring demo...")
    
    for i in range(20):  # 20 predictions
        try:
            # Pick random sample
            sample = random.choice(test_samples)
            
            # Make prediction
            response = requests.post(f"{base_url}/predict", json=sample, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction {i+1}: Disease Risk {result['disease_risk_prediction']}, "
                      f"Cholesterol {result['cholesterol_prediction']:.1f}")
            else:
                print(f"❌ Prediction {i+1} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request {i+1} error: {e}")
        
        # Wait between requests
        time.sleep(2)
    
    print("🎉 Traffic generation completed!")

if __name__ == "__main__":
    generate_traffic()