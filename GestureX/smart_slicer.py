import cv2
import numpy as np
import os
import pandas as pd
import mediapipe as mp

# Direct imports for stability
try:
    import mediapipe.solutions.hands as mp_hands
except ImportError:
    from mediapipe.python.solutions import hands as mp_hands

# --- CONFIG ---
CHART_DIR = "charts"
DEBUG_DIR = "extracted_debug"
# --------------

def slice_and_extract():
    hands_solution = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=2, 
        min_detection_confidence=0.1 # Very low to catch sketches
    )
    
    if not os.path.exists(DEBUG_DIR): os.makedirs(DEBUG_DIR)

    images = [f for f in os.listdir(CHART_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, img_name in enumerate(images[:10]): # Scan first 10 for variety
        path = os.path.join(CHART_DIR, img_name)
        img = cv2.imread(path)
        if img is None: continue
        
        h, w, _ = img.shape
        # More flexible grid for high-res images
        rows, cols = 6, 6 
        cell_h, cell_w = h // rows, w // cols
        
        lang_code = f"LANG_{i+1}"
        print(f"📌 Analyzing {img_name}...")
        
        data_rows = []
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * cell_h, min((r+1) * cell_h, h)
                x1, x2 = c * cell_w, min((c+1) * cell_w, w)
                cell = img[y1:y2, x1:x2]
                if cell.size == 0: continue
                
                # Try multiple enhancements
                for alpha in [1.2, 1.8]:
                    enhanced = cv2.convertScaleAbs(cell, alpha=alpha, beta=10)
                    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                    results = hands_solution.process(rgb)
                    
                    if results.multi_hand_landmarks:
                        # Save for debug
                        debug_path = os.path.join(DEBUG_DIR, f"{lang_code}_{r}_{c}.jpg")
                        cv2.imwrite(debug_path, cell)
                        
                        def get_73(hand_lms):
                            lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
                            lms = lms - lms[0]
                            dist = np.linalg.norm(lms[9]-lms[0])
                            if dist > 0: lms = lms / dist
                            f = lms.flatten()
                            tips = [4, 8, 12, 16, 20]
                            ds = [np.linalg.norm(lms[tips[i]]-lms[tips[j]]) for i in range(len(tips)) for j in range(i+1, len(tips))]
                            return np.concatenate([f, ds])
                        
                        label = f"CHAR_{r}_{c}"
                        num = len(results.multi_hand_landmarks)
                        if num == 1:
                            feat = get_73(results.multi_hand_landmarks[0])
                            data_rows.append(list(feat) + [label, lang_code, "single"])
                        elif num >= 2:
                            f1 = get_73(results.multi_hand_landmarks[0])
                            f2 = get_73(results.multi_hand_landmarks[1])
                            data_rows.append(list(np.concatenate([f1, f2])) + [label, lang_code, "dual"])
                        
                        break # Found, skip other enhancements

        if data_rows:
            csv_path = f"data/{lang_code}_gestures.csv"
            if not os.path.exists("data"): os.makedirs("data")
            max_f = len(data_rows[0]) - 3
            cols_names = [f"f{i}" for i in range(max_f)] + ["label", "language", "hand_type"]
            df = pd.DataFrame(data_rows, columns=cols_names)
            df.to_csv(csv_path, index=False)
            print(f"✅ SUCCESS: {lang_code} model data saved ({len(df)} samples)")

if __name__ == "__main__":
    slice_and_extract()
