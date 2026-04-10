import pandas as pd
import numpy as np
import os
import joblib
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- CONFIGURE THIS ---
DATA_PATH = "gesture_data_augmented.csv"
MODEL_PATH = "word_model.onnx"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
OUTPUT_DIR = "eval_results"
# ----------------------

def run_evaluation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load resources
    engine = ort.InferenceSession(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Check for language column, default to 'Universal' if missing
    group_col = 'language' if 'language' in df.columns else 'label'
    
    results = []
    
    print(f"\nEvaluating system performance grouped by: {group_col}")
    
    # 2. Iterate through groups
    unique_groups = df[group_col].unique()
    
    for group in unique_groups:
        group_df = df[df[group_col] == group]
        X = group_df.iloc[:, :-1].values
        y_true_labels = group_df['label'].values
        y_true = le.transform(y_true_labels)
        
        # Inference
        X_scaled = scaler.transform(X).astype(np.float32)
        input_name = engine.get_inputs()[0].name
        # ONNX scikit-learn output [label, probabilities]
        raw_preds = engine.run(None, {input_name: X_scaled})[0]
        y_pred = raw_preds
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Finding confused pairs
        cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
        confusions = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:
                    confusions.append((le.classes_[i], le.classes_[j], cm[i][j]))
        
        confusions = sorted(confusions, key=lambda x: x[2], reverse=True)[:1]
        weakest = f"{confusions[0][0]} -> {confusions[0][1]}" if confusions else "None"
        
        results.append({
            "Group": group,
            "Accuracy": f"{acc*100:.1f}%",
            "F1-Score": f"{f1:.2f}",
            "Weakest Pair": weakest
        })
        
        # Save per-group confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Greens')
        plt.title(f"Confusion Matrix: {group}")
        plt.savefig(f"{OUTPUT_DIR}/cm_{group}.png")
        plt.close()

    # 3. Output Table
    report_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print(f"{'Group':<15} | {'Accuracy':<10} | {'F1-Score':<10} | {'Weakest Pair'}")
    print("-" * 60)
    for index, row in report_df.iterrows():
        print(f"{row['Group']:<15} | {row['Accuracy']:<10} | {row['F1-Score']:<10} | {row['Weakest Pair']}")
    print("="*60)
    
    # Check for low performers
    for index, row in report_df.iterrows():
        if float(row['F1-Score']) < 0.80:
            print(f"⚠️  ALERT: {row['Group']} is underperforming. Needs more training data.")

    print(f"\n✅ Evaluation complete. Group matrices saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_evaluation()
