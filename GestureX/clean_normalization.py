import pandas as pd
import numpy as np
import os

# --- CONFIGURE THIS ---
INPUT_PATH = "data/landmarks/ASL/word_data.csv"
OUTPUT_PATH = "gesture_data_clean.csv"
# ----------------------

def clean_and_rescale():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    rows_before = len(df)
    
    clean_rows = []
    dropped_count = 0

    print("\n--- NORMALIZATION & RE-SCALING ---")
    
    for _, row in df.iterrows():
        # f0, f1, f2 are wrist x, y, z
        landmarks = row.iloc[:-1].values.astype(float).reshape(21, 3)
        label = row['label']
        
        # 1. Wrist Centering: Subtract wrist (point 0) from all points
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist
        
        # 2. Re-scaling: Distance between Wrist(0) and Middle MCP(9)
        dist = np.linalg.norm(landmarks[0] - landmarks[9])
        
        # 3. Drop failed rows
        if dist == 0 or np.all(landmarks == 0):
            dropped_count += 1
            continue
            
        landmarks = landmarks / dist
        
        # Flatten back
        new_row = dict(zip(df.columns[:-1], landmarks.flatten()))
        new_row['label'] = label
        clean_rows.append(new_row)

    final_df = pd.DataFrame(clean_rows)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Original rows: {rows_before}")
    print(f"Rows dropped: {dropped_count}")
    print(f"Final clean rows: {len(final_df)}")
    print(f"✅ Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_and_rescale()
