import pandas as pd
import numpy as np
import os

# --- CONFIGURE THIS ---
DATA_PATH = "data/landmarks/ASL/word_data.csv"
# ----------------------

def run_diagnostic():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)
    
    print("\n" + "="*40)
    print("📋 EMERGENCY DATA DIAGNOSTIC REPORT")
    print("="*40)

    # 1. Class Distribution Check
    counts = df['label'].value_counts().sort_values()
    avg_count = counts.mean()
    
    print("\n--- 1. Class Distribution ---")
    critical_classes = []
    dominant_classes = []
    
    for label, count in counts.items():
        status = ""
        if count < 150:
            status = " | [CRITICAL]"
            critical_classes.append(label)
        elif count > (3 * avg_count):
            status = " | [DOMINANT]"
            dominant_classes.append(label)
        print(f"{label:<15}: {count}{status}")

    # 2. Data Source Separation (Mock split for audit)
    # Split row 0 to N/2 as 'Source A' and rest as 'Source B'
    half = total_rows // 2
    source_a = df.iloc[:half]
    source_b = df.iloc[half:]
    print(f"\n--- 2. Data Source Split (Assumed) ---")
    print(f"Source A (0-{half}): {len(source_a)} rows")
    print(f"Source B ({half}-{total_rows}): {len(source_b)} rows")

    # 3. Feature Distribution Check (Collision Risk)
    print("\n--- 3. Collision Risk (Gesture Confusion) ---")
    collisions = []
    # Use first 10 columns (f0 to f9)
    features = df.columns[:-1][:10]
    class_means = df.groupby('label')[features].mean()
    
    labels = list(class_means.index)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            mean_diff = np.abs(class_means.loc[l1] - class_means.loc[l2]).mean()
            if mean_diff < 0.05:
                print(f"⚠️  COLLISION RISK: '{l1}' and '{l2}' patterns are too similar (diff: {mean_diff:.4f})")
                collisions.append((l1, l2))

    # 4. Normalization Consistency Check
    print("\n--- 4. Normalization Consistency ---")
    # Wrist is f0, f1, f2
    wrist_cols = ['f0', 'f1', 'f2']
    if all(c in df.columns for c in wrist_cols):
        unnormalized = df[(df[wrist_cols].abs() > 0.1).any(axis=1)]
        unnormalized_count = len(unnormalized)
        if unnormalized_count > 0:
            print(f"❌ NOT NORMALIZED: Found {unnormalized_count} rows where wrist is not at origin.")
        else:
            print("✅ Normalization: All rows are wrist-centered.")
    else:
        print("⚠️ Warning: f0, f1, f2 columns not found to verify wrist origin.")
        unnormalized_count = 0

    # 5. Final Verdict
    print("\n" + "="*40)
    print("🏁 FINAL VERDICT")
    print("="*40)
    print(f"Total Samples           : {total_rows}")
    print(f"Critical Classes        : {len(critical_classes)}")
    print(f"Dominant Classes        : {len(dominant_classes)}")
    print(f"Collision Risk Pairs    : {len(collisions)}")
    print(f"Unnormalized Rows       : {unnormalized_count}")
    
    print("\n👉 ACTION PLAN:")
    if unnormalized_count > 0:
        print("1. RUN PROMPT 2B: You must fix the normalization mismatch first.")
    if len(dominant_classes) > 0 or len(critical_classes) > 0:
        print("2. RUN PROMPT 2A: Rebalance your classes to stop 'YES' from dominating 'HELLO'.")
    if len(collisions) > 0:
        print("3. RUN PROMPT 2C: Use inter-finger distance features to separate similar gestures.")

if __name__ == "__main__":
    run_diagnostic()
