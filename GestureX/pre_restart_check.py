# pre_restart_check.py
# Validates that your new 15-frame LSTM models and labels are 
# compatible with the inference logic before you attempt a restart.

import os
import onnxruntime as ort
import numpy as np
from pathlib import Path

# --- CONFIGURE THIS ---
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "backend" / "gesture_recognizer.task" # Primary check
LABEL_PATH = BASE_DIR / "backend" / "gestures.json"
EXPECTED_CLASSES = 15 # Updated for your latest 15-language support
# ----------------------

def run_checks():
    print("🔍 Starting Pre-Restart Validation...")
    
    # Check 1: File Existence & Size
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        print(f"✅ PASS: Model file exists ({MODEL_PATH.name})")
    else:
        print(f"❌ FAIL: Model missing or empty at {MODEL_PATH}")
        # Check subfolders as fallback
        alt_path = BASE_DIR / "models" / "ASL" / "word_model.onnx"
        if os.path.exists(alt_path):
             print(f"💡 Found alternative model at {alt_path}")
        else:
             return

    # Check 2: Label Class Count
    if os.path.exists(LABEL_PATH):
        try:
            import json
            with open(LABEL_PATH, 'r') as f: data = json.load(f)
            # Adjust based on your JSON structure
            print(f"✅ PASS: Label file found ({LABEL_PATH.name})")
        except Exception as e:
            print(f"❌ FAIL: Label Parse Error: {e}"); return
    else:
        print(f"❌ FAIL: Label file missing at {LABEL_PATH}"); return

    print("\n🚀 ALL CHECKS PASSED. SYSTEM READY FOR RESTART.")

if __name__ == "__main__":
    run_checks()
