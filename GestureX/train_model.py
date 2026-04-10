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
from skl2onnx.common.data_types import FloatTensorType

# --- CONFIGURE THIS ---
DATA_PATH = "gesture_data_augmented.csv"
MODEL_ONNX_PATH = "word_model.onnx"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
# ----------------------

def train_best_model():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1].values
    y_raw = df['label'].values
    
    # 1. Preprocessing
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
    
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 3. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_preds, average='weighted')
    print(f"Random Forest Weighted F1: {rf_f1:.4f}")
    
    # 4. MLP Neural Network
    print("\nTraining MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu', 
        early_stopping=True, 
        random_state=42,
        max_iter=500
    )
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_f1 = f1_score(y_test, mlp_preds, average='weighted')
    print(f"MLP Weighted F1: {mlp_f1:.4f}")
    
    # 5. Pick the winner
    best_model = None
    if rf_f1 >= mlp_f1:
        print("\n🏆 Random Forest wins!")
        best_model = rf
        best_preds = rf_preds
    else:
        print("\n🏆 MLP wins!")
        best_model = mlp
        best_preds = mlp_preds
        
    # 6. Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds, target_names=le.classes_))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title("Best Model Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("best_model_confusion_matrix.png")
    
    # 7. Export to ONNX
    print(f"\nExporting best model to {MODEL_ONNX_PATH}...")
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    
    # Note: MLP and RF need different conversion paths in some older skl2onnx versions,
    # but to_onnx handles most scikit-learn estimators automatically.
    onx = to_onnx(best_model, X_train[:1].astype(np.float32))
    with open(MODEL_ONNX_PATH, "wb") as f:
        f.write(onx.SerializeToString())
        
    print("✅ Training and Export complete!")

if __name__ == "__main__":
    train_best_model()
