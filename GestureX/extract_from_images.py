import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# --- CONFIG ---
IMAGE_FOLDER = "training_images"
# --------------

def extract_from_images(lang_code):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    
    lang_path = os.path.join(IMAGE_FOLDER, lang_code)
    if not os.path.exists(lang_path):
        print(f"❌ Error: Folder {lang_path} not found.")
        return

    data_rows = []
    print(f"📷 Scanning {lang_code} images...")

    for filename in os.listdir(lang_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
        # Determine label from filename (e.g., "A.jpg" -> "A")
        label = os.path.splitext(filename)[0].upper()
        
        img_path = os.path.join(lang_path, filename)
        image = cv2.imread(img_path)
        if image is None: continue
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # Helper: construct 73 features
            def get_73(hand_lms):
                lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
                lms = lms - lms[0] # Wrist center
                dist = np.linalg.norm(lms[9] - lms[0])
                if dist > 0: lms = lms / dist
                flat = lms.flatten()
                tips = [4, 8, 12, 16, 20]
                dists = [np.linalg.norm(lms[tips[i]] - lms[tips[j]]) for i in range(len(tips)) for j in range(i+1, len(tips))]
                return np.concatenate([flat, dists])

            if num_hands == 1:
                features = get_73(results.multi_hand_landmarks[0])
                data_rows.append(list(features) + [label, lang_code, "single"])
            elif num_hands == 2:
                f1 = get_73(results.multi_hand_landmarks[0])
                f2 = get_73(results.multi_hand_landmarks[1])
                features = np.concatenate([f1, f2])
                data_rows.append(list(features) + [label, lang_code, "dual"])
            
            print(f"✅ Processed {label} (Found {num_hands} hand(s))")
        else:
            print(f"⚠️ Failed to find hand in {filename}. Ensure lighting is clear.")

    if data_rows:
        csv_path = f"data/{lang_code}_gestures.csv"
        max_f = len(data_rows[0]) - 3
        cols = [f"f{i}" for i in range(max_f)] + ["label", "language", "hand_type"]
        df = pd.DataFrame(data_rows, columns=cols)
        df.to_csv(csv_path, index=False)
        print(f"\n✨ DONE: Extracted {len(df)} samples to {csv_path}")
    else:
        print("No landmarks were found in any images.")

if __name__ == "__main__":
    code = input("Enter language code to scan (e.g., BSL, KSL): ").upper()
    extract_from_images(code)
