# DL_MLOPS/test_api.py
"""
Simple test client for the FastAPI trading API.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

print("="*60)
print("TESTING API")
print("="*60)

# 1️⃣ Health check
print("\n[1/3] Health check...")
response = requests.get(f"{BASE_URL}/")
print("✅", response.json())

# 2️⃣ Model info
print("\n[2/3] Model info...")
response = requests.get(f"{BASE_URL}/model_info")
print("✅", response.json())

# 3️⃣ Prediction test
print("\n[3/3] Test prediction...")

# Fake sequence (30 timesteps x 29 features)
sample_sequence = [[0.1]*29 for _ in range(30)]

payload = {
    "sequence": sample_sequence,
    "long_threshold": 0.55,
    "short_threshold": 0.55
}

response = requests.post(f"{BASE_URL}/predict", json=payload)

if response.status_code == 200:
    print("✅ Prediction result:")
    print(response.json())
else:
    print("❌ Error:", response.status_code)
    print(response.json())

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("For interactive docs: http://localhost:8000/docs")
print("="*60)
