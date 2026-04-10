import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from skl2onnx import to_onnx

# --- CONFIGURE THIS ---
DATA_PATH = "gesture_data_enriched.csv"
MODEL_ONNX_PATH = "word_model_v2.onnx"
SCALER_PATH = "scaler_v2.pkl"
ENCODER_PATH = "label_encoder_v2.pkl"
# ----------------------

def train_v2():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1].values
    y_raw = df['label'].values
    
    # 1. Encoding & Scaling
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    joblib.dump(le, ENCODER_PATH)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    
    # 2. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Random Forest
    print("\nTraining RandomForest (300 estimators)...")
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_f1 = f1_score(y_test, rf.predict(X_test), average='weighted')
    
    # 4. MLP
    print("Training MLP (256, 128, 64)...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True)
    mlp.fit(X_train, y_train)
    mlp_f1 = f1_score(y_test, mlp.predict(X_test), average='weighted')
    
    # 5. Pick Winner
    best_model = rf if rf_f1 >= mlp_f1 else mlp
    winner_name = "RandomForest" if best_model == rf else "MLP"
    print(f"\n🏆 Winner: {winner_name} (F1: {max(rf_f1, mlp_f1):.4f})")
    
    # 6. Evaluation
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    print("\nInvididual Class Performance:")
    for label, metrics in report.items():
        if label in le.classes_:
            f1 = metrics['f1-score']
            flag = "⚠️ [LOW PERFORMANCE]" if f1 < 0.75 else "✅"
            print(f"{label:<12}: {f1:.4f} {flag}")
            
    # 7. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Purples')
    plt.title("Model V2 Confusion Matrix (Enriched Features)")
    plt.savefig("confusion_matrix_v2.png")
    
    # 8. Export to ONNX
    print(f"\nExporting to {MODEL_ONNX_PATH}...")
    onx = to_onnx(best_model, X_train[:1].astype(np.float32))
    with open(MODEL_ONNX_PATH, "wb") as f:
        f.write(onx.SerializeToString())
        
    print("✨ All systems complete. Use word_model_v2.onnx in your app.")

if __name__ == "__main__":
    train_v2()
