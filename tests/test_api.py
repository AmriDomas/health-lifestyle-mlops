# tests/test_api.py
import requests
import json
import pytest
from typing import Dict, Any, List
import pandas as pd
import time

class HealthAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def test_health_check(self) -> bool:
        """Test endpoint health check"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed. Models loaded: {health_data.get('models_loaded', False)}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Make sure the server is running.")
            return False
    
    def test_debug_shap(self) -> bool:
        """Test debug SHAP endpoint"""
        try:
            response = requests.get(f"{self.base_url}/debug-shap")
            if response.status_code == 200:
                shap_data = response.json()
                print(f"✅ Debug SHAP successful")
                print(f"   Features: {len(shap_data.get('features', []))}")
                print(f"   SHAP values: {len(shap_data.get('shap_values', []))}")
                return True
            else:
                print(f"❌ Debug SHAP failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Debug SHAP error: {e}")
            return False
    
    def create_test_payload(self, **kwargs) -> Dict[str, Any]:
        """Create test payload dengan default values yang sesuai Streamlit"""
        default_data = {
            "age": 30,
            "gender": "Male",
            "bmi": 23.0,
            "daily_steps": 5000,  # ✅ Diubah dari float ke int
            "sleep_hours": 8.0,
            "water_intake_l": 2.0,
            "calories_consumed": 2000,  # ✅ Diubah dari float ke int
            "smoker": 0,
            "alcohol": 0,
            "resting_hr": 60,  # ✅ Diubah dari float ke int
            "systolic_bp": 120,  # ✅ Diubah dari float ke int
            "diastolic_bp": 80,  # ✅ Diubah dari float ke int
            "family_history": 0
        }
        
        # Update dengan values yang diberikan
        default_data.update(kwargs)
        return default_data
    
    def test_single_prediction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Test single prediction endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction successful for age {payload.get('age')}")
                return result
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def validate_prediction_response(self, result: Dict[str, Any]) -> bool:
        """Validate structure of prediction response"""
        required_fields = [
            "disease_risk_prediction", 
            "cholesterol_prediction", 
            "cluster_assignment"
        ]
        
        optional_fields = ["confidence", "feature_importance", "shap_values"]
        
        # Check required fields
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate data types
        try:
            disease_risk = result["disease_risk_prediction"]
            cholesterol = float(result["cholesterol_prediction"])
            cluster = int(result["cluster_assignment"])
            
            print(f"   Disease Risk: {disease_risk}")
            print(f"   Cholesterol: {cholesterol:.2f}")
            print(f"   Cluster: {cluster}")
            
            # Check optional fields
            if "confidence" in result and result["confidence"] is not None:
                confidence = float(result["confidence"])
                print(f"   Confidence: {confidence:.2%}")
            
            if "feature_importance" in result and result["feature_importance"]:
                fi_count = len(result["feature_importance"])
                print(f"   Feature Importance: {fi_count} features")
            
            if "shap_values" in result and result["shap_values"]:
                shap_count = len(result["shap_values"])
                print(f"   SHAP Values: {shap_count} values")
            
            return True
            
        except (ValueError, TypeError) as e:
            print(f"❌ Data type validation failed: {e}")
            return False
    
    def test_edge_cases(self):
        """Test berbagai edge cases"""
        edge_cases = [
            # Normal case
            ("Normal case", self.create_test_payload(age=25, bmi=22.0)),
            # Extreme values
            ("Young healthy", self.create_test_payload(age=18, bmi=18.5, smoker=0, alcohol=0)),
            ("Elderly high risk", self.create_test_payload(age=80, bmi=35.0, smoker=1, alcohol=1)),
            # Different categories
            ("Female smoker", self.create_test_payload(gender="Female", smoker=1)),
            ("Family history", self.create_test_payload(family_history=1)),
            # Boundary values
            ("Min values", self.create_test_payload(age=18, bmi=15.0, daily_steps=1000)),
            ("Max values", self.create_test_payload(age=80, bmi=40.0, daily_steps=20000)),
        ]
        
        print("🧪 Testing Edge Cases...")
        results = []
        
        for case_name, payload in edge_cases:
            print(f"\n🔬 Case: {case_name}")
            print(f"   Payload: { {k: v for k, v in payload.items() if k in ['age', 'bmi', 'smoker', 'alcohol']} }")
            
            result = self.test_single_prediction(payload)
            if result:
                is_valid = self.validate_prediction_response(result)
                if is_valid:
                    results.append((case_name, True))
                else:
                    results.append((case_name, False))
            else:
                results.append((case_name, False))
            
            time.sleep(0.5)  # Delay antara requests
        
        # Summary
        print("\n📊 Edge Cases Summary:")
        for case_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} - {case_name}")
        
        return all(success for _, success in results)
    
    def test_performance(self, num_requests: int = 5):
        """Test performance dengan multiple requests"""
        print(f"\n⚡ Performance Testing ({num_requests} requests)...")
        
        test_cases = [self.create_test_payload(age=25 + i*5) for i in range(num_requests)]
        response_times = []
        successful_requests = 0
        
        for i, payload in enumerate(test_cases):
            start_time = time.time()
            result = self.test_single_prediction(payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if result and self.validate_prediction_response(result):
                successful_requests += 1
                print(f"   Request {i+1}: {response_time:.2f}s ✅")
            else:
                print(f"   Request {i+1}: {response_time:.2f}s ❌")
            
            time.sleep(0.2)  # Small delay antara requests
        
        # Performance summary
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            success_rate = (successful_requests / num_requests) * 100
            
            print(f"\n📈 Performance Summary:")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Average Response Time: {avg_time:.2f}s")
            print(f"   Min Response Time: {min_time:.2f}s")
            print(f"   Max Response Time: {max_time:.2f}s")
            
            return success_rate >= 80  # Consider successful if 80%+ requests pass
        else:
            return False

def main():
    """Main function untuk testing API"""
    tester = HealthAPITester()
    
    print("🚀 Starting Health API Tests...")
    print("=" * 60)
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    health_ok = tester.test_health_check()
    
    if not health_ok:
        print("❌ Health check failed. Exiting tests.")
        return
    
    # Test 2: Debug SHAP
    print("\n2. Testing debug SHAP endpoint...")
    shap_ok = tester.test_debug_shap()
    
    # Test 3: Single Prediction dengan data dari Streamlit
    print("\n3. Testing single prediction...")
    
    streamlit_payload = tester.create_test_payload(
        age=30,
        gender="Male", 
        bmi=23.0,
        daily_steps=5000,
        sleep_hours=8.0,
        water_intake_l=2.0,
        calories_consumed=2000,
        smoker=0,
        alcohol=0,
        resting_hr=60,
        systolic_bp=120,
        diastolic_bp=80,
        family_history=0
    )
    
    result = tester.test_single_prediction(streamlit_payload)
    prediction_ok = False
    if result:
        prediction_ok = tester.validate_prediction_response(result)
    
    # Test 4: Edge Cases
    print("\n4. Testing edge cases...")
    edge_cases_ok = tester.test_edge_cases()
    
    # Test 5: Performance Testing
    print("\n5. Performance testing...")
    performance_ok = tester.test_performance(num_requests=3)
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎯 FINAL TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Health Check", health_ok),
        ("Debug SHAP", shap_ok),
        ("Single Prediction", prediction_ok),
        ("Edge Cases", edge_cases_ok),
        ("Performance", performance_ok)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    # Run tests
    success = main()
    
    # Exit dengan code yang appropriate untuk CI/CD
    exit(0 if success else 1)