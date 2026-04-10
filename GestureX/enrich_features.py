import pandas as pd
import numpy as np
import os

# --- CONFIGURE THIS ---
INPUT_PATH = "gesture_data_clean.csv"
OUTPUT_PATH = "gesture_data_enriched.csv"
# ----------------------

def enrich_features():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    print("\n--- FEATURE ENRICHMENT (INTER-FINGER DISTANCES) ---")
    
    # MediaPipe Tip Indices:
    # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
    tips = [4, 8, 12, 16, 20]
    
    enriched_data = []
    
    for _, row in df.iterrows():
        # f0, f1, f2 ... f62
        landmarks = row.iloc[:-1].values.astype(float).reshape(21, 3)
        label = row['label']
        
        # Calculate 10 distances between fingertips
        new_features = {}
        dist_idx = 1
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                p1 = landmarks[tips[i]]
                p2 = landmarks[tips[j]]
                d = np.linalg.norm(p1 - p2)
                new_features[f'd{dist_idx}'] = d
                dist_idx += 1
        
        # Merge original features + new distances + label
        combined = {**dict(row.iloc[:-1]), **new_features, "label": label}
        enriched_data.append(combined)

    final_df = pd.DataFrame(enriched_data)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Original features: 63")
    print(f"New distance features: 10")
    print(f"Total features: 73")
    print(f"✅ Samples enriched: {len(final_df)}")
    print(f"✅ Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    enrich_features()
