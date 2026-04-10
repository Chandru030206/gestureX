import pandas as pd
import numpy as np
import os

# --- CONFIGURE THIS ---
INPUT_PATH = "data/landmarks/ASL/word_data.csv"
OUTPUT_PATH = "gesture_data_augmented.csv"
# ----------------------

def augment_data():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    counts = df['label'].value_counts()
    
    print("\n--- DATA AUGMENTATION ---")
    print(f"Original samples: {len(df)}")
    
    augmented_rows = []
    
    for label, count in counts.items():
        class_df = df[df['label'] == label]
        augmented_rows.append(class_df) # Keep original
        
        target = 300
        needed = target - count
        
        if needed > 0:
            print(f"Generating {needed} augmented samples for class: {label}")
            
            # Generate needed samples using random original samples
            temp_list = []
            for _ in range(needed):
                # Pick a random sample from the original class data
                sample = class_df.sample(1).iloc[0].copy()
                landmarks = sample.iloc[:-1].values.astype(float)
                
                # Randomly choose an augmentation technique
                tech = np.random.choice(['flip', 'noise', 'scale', 'rotate'])
                
                if tech == 'flip':
                    # f0, f3, f6... are X coordinates
                    landmarks[::3] = -landmarks[::3]
                elif tech == 'noise':
                    landmarks += np.random.normal(0, 0.01, landmarks.shape)
                elif tech == 'scale':
                    landmarks *= np.random.uniform(0.88, 1.12)
                elif tech == 'rotate':
                    # Small 2D rotation for X (index 0, 3...) and Y (index 1, 4...)
                    angle = np.radians(np.random.uniform(-8, 8))
                    c, s = np.cos(angle), np.sin(angle)
                    for i in range(0, 63, 3):
                        x, y = landmarks[i], landmarks[i+1]
                        landmarks[i] = x * c - y * s
                        landmarks[i+1] = x * s + y * c
                
                # Create a new augmented row
                new_row = dict(zip(df.columns[:-1], landmarks))
                new_row['label'] = label
                temp_list.append(new_row)
            
            augmented_rows.append(pd.DataFrame(temp_list))

    final_df = pd.concat(augmented_rows).sample(frac=1).reset_index(drop=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Augmentation complete.")
    print(f"New samples count: {len(final_df)}")
    print(f"Final Class Counts:\n{final_df['label'].value_counts()}")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    augment_data()
