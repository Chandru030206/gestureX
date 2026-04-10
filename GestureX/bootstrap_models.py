import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import to_onnx

# --- ASL/BSL/ISL GEOMETRY CONSTANTS ---
# We simulate the 10 finger-distances for common signs (A=closed, B=flat, etc.)
# --------------------------------------

def generate_synthetic_and_train(lang_code, gestures):
    print(f"🧬 Bootstrapping {lang_code} model from geometric constants...")
    
    data = []
    for label in gestures:
        # Create 100 samples per gesture with variation
        for _ in range(100):
            # Base 63 landmarks (simulated)
            lms = np.random.normal(0, 0.05, (21, 3))
            lms = lms - lms[0] # origin
            
            # Simulated 10 distances (d1-d10) based on label profile
            # Case 1: Closed Hand (A, S, T) - small distances
            if label in ['A', 'S', 'T', 'NO', 'YES']:
                dists = np.random.uniform(0.05, 0.15, 10)
            # Case 2: Open Hand (B, HELLO, THANK YOU) - large distances
            else:
                dists = np.random.uniform(0.4, 0.9, 10)
                
            row = list(lms.flatten()) + list(dists) + [label, lang_code, "single"]
            data.append(row)

    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(73)] + ["label", "language", "hand_type"])
    
    # Train
    X = df.iloc[:, :-3].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_s, y)
    
    # Save
    path = f"models/{lang_code}"
    if not os.path.exists(path): os.makedirs(path)
    joblib.dump(scaler, f"{path}/scaler.pkl")
    joblib.dump(le, f"{path}/label_encoder.pkl")
    
    # ONNX
    onx = to_onnx(rf, X_s[:1])
    with open(f"{path}/word_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    
    # Info
    with open(f"{path}/model_info.json", "w") as f:
        json.dump({"language": lang_code, "hand_type": "single", "classes": gestures}, f)

if __name__ == "__main__":
    languages = {
        "BSL": ["HELLO", "THANK YOU", "YES", "NO", "A", "B", "C"],
        "ISL": ["HELLO", "THANK YOU", "YES", "NO", "NAMASTE"],
        "JSL": ["HELLO", "THANK YOU", "ARIGATO"],
        "KSL": ["HELLO", "SARANGHAE"],
        "CSL": ["HELLO", "NI HAO"]
    }
    for lang, gs in languages.items():
        generate_synthetic_and_train(lang, gs)
    print("✨ ALL REGIONAL MODELS BOOTSTRAPPED SUCCESSFULLY.")
