# smoke_test.py
# Sends a specific high-fidelity sequence (zeros)
# to the live API to ensure the backend accepts the input shape.

import requests
import json
import numpy as np

# --- CONFIGURE THIS ---
API_URL = "http://127.0.0.1:8000/predict_gesture"
# ----------------------

def run_smoke_test():
    print("🧪 Running Post-Restart Smoke Test...")
    
    # Payload simulating a frame from the frontend
    # Note: Backend expects 'image' (base64) usually, but we check if it handles data.
    payload = {
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==" # 1x1 black dot
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # The result might be 'No hand detected' since we sent a black dot
            print(f"✅ SMOKE TEST PASSED (Server Responded)")
            print(f"Server Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ SMOKE TEST FAILED: Status {response.status_code}")
            print(f"Body: {response.text}")
            
    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")

if __name__ == "__main__":
    run_smoke_test()
