import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
from skl2onnx import to_onnx
from datetime import datetime

# --- CONFIG ---
DATA_DIR = "data"
MODELS_ROOT = "models"
# --------------

def train_language(lang_code):
    csv_path = os.path.join(DATA_DIR, f"{lang_code}_gestures.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n--- Training Engine: {lang_code} ---")
    
    # 1. Split Single/Dual (The model needs fixed input sizes)
    hand_types = df['hand_type'].unique()
    for h_type in hand_types:
        subset = df[df['hand_type'] == h_type]
        print(f"\nTraining {h_type}-hand model branch for {lang_code}...")
        
        counts = subset['label'].value_counts()
        print(f"Samples per label:\n{counts}")
        
        # 2. Quality Checks
        if subset['label'].nunique() < 2:
            print("Skipping: Need at least 2 categories to train.")
            continue
            
        # 3. Augmentation (Gaussian Noise + Mirror)
        # For simplicity, we sample up if below 300
        balanced_list = []
        target = 300
        for label, count in counts.items():
            label_df = subset[subset['label'] == label]
            balanced_list.append(label_df)
            if count < target:
                needed = target - count
                upsampled = label_df.sample(needed, replace=True).copy()
                # Landmark data is everything except last 3 cols
                lms_only = upsampled.iloc[:, :-3]
                noise = np.random.normal(0, 0.012, lms_only.shape)
                upsampled.iloc[:, :-3] = lms_only + noise
                balanced_list.append(upsampled)
        
        final_subset = pd.concat(balanced_list).sample(frac=1).reset_index(drop=True)
        
        # 4. Preparation
        X = final_subset.iloc[:, :-3].values.astype(np.float32)
        y_raw = final_subset['label'].values
        
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
        
        # 5. Hybrid Model Selection
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        rf_f1 = f1_score(y_test, rf.predict(X_test), average='weighted')
        
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True)
        mlp.fit(X_train, y_train)
        mlp_f1 = f1_score(y_test, mlp.predict(X_test), average='weighted')
        
        best = rf if rf_f1 >= mlp_f1 else mlp
        winner_name = "RandomForest" if best == rf else "MLP"
        f1_final = max(rf_f1, mlp_f1)
        
        print(f"🏆 Best Model: {winner_name} (F1: {f1_final:.4f})")
        
        # 6. Save Artifacts
        lang_model_dir = os.path.join(MODELS_ROOT, lang_code)
        if not os.path.exists(lang_model_dir): os.makedirs(lang_model_dir)
        # 7. Confusion Matrix (Skipped in headless mode)
        # plt.figure(figsize=(10, 8))
        # cm = confusion_matrix(y_test, y_pred)
        # sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Purples')
        # plt.title(f"{lang_code} Confusion Matrix")
        # plt.savefig(os.path.join(lang_model_dir, "confusion_matrix.png"))
        
        # ONNX Export
        onx = to_onnx(best, X_train[:1])
        model_name = "word_model.onnx" if h_type == "single" else "word_model_dual.onnx"
        with open(os.path.join(lang_model_dir, model_name), "wb") as f:
            f.write(onx.SerializeToString())
            
        joblib.dump(scaler, os.path.join(lang_model_dir, "scaler.pkl"))
        joblib.dump(le, os.path.join(lang_model_dir, "label_encoder.pkl"))
        
        # Model Info
        info = {
            "language": lang_code,
            "classes": list(le.classes_),
            "hand_type": h_type,
            "f1_score": f1_final,
            "feature_count": X.shape[1],
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        with open(os.path.join(lang_model_dir, "model_info.json"), "w") as f:
            json.dump(info, f, indent=4)
            
        print(f"✅ Training Artifacts saved to {lang_model_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        lang_to_train = sys.argv[1].upper()
    else:
        lang_to_train = input("Enter language code to train (e.g., ASL): ").upper()
    
    train_language(lang_to_train)
