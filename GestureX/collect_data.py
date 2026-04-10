import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time

# --- CONFIGURATION ---
DATA_DIR = "data"
# ---------------------

class GestureCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # User Inputs
        self.lang_code = input("Enter Language Code (e.g., BSL, ISL, JSL): ").upper()
        self.gesture_label = input("Enter Gesture Label (e.g., HELLO, A): ").upper()
        self.target_samples = int(input("Samples to collect (default 300): ") or 300)
        
        self.collected_count = 0
        self.is_collecting = False
        self.data_rows = []
        self.csv_path = os.path.join(DATA_DIR, f"{self.lang_code}_gestures.csv")
        
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

    def _get_73_features(self, hand_landmarks):
        """Constructs 73-feature vector for ONE hand."""
        lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # 1. Wrist Normalization
        wrist = lms[0].copy()
        lms = lms - wrist
        
        # 2. Unit Scaling (Wrist to Middle MCP index 9)
        dist = np.linalg.norm(lms[0] - lms[9])
        if dist > 0: lms = lms / dist
        
        # 3. Flatten (63)
        flat = lms.flatten()
        
        # 4. Finger Tip Distances (10)
        tips = [4, 8, 12, 16, 20]
        distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                distances.append(np.linalg.norm(lms[tips[i]] - lms[tips[j]]))
        
        return np.concatenate([flat, distances])

    def start(self):
        # Try Camera 0, then Camera 1 (for some Mac setups)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Camera 0 failed, trying Camera 1...")
            cap = cv2.VideoCapture(1)
            
        if not cap.isOpened():
            print("❌ ERROR: Could not open any camera.")
            print("Please check: System Settings > Privacy & Security > Camera")
            return

        print("\n--- CONTROLS ---")
        print("SPACE: Start/Pause | R: Reset Batch | Q: Save & Quit")
        
        # Countdown with safety check
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.putText(frame, f"Starting in {i}...", (200, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow("GestureX Collector", frame)
                cv2.waitKey(1000)
            else:
                print(f"Waiting for camera... {i}")
                time.sleep(1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            hand_type = "single"
            features = None
            
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                
                # Draw Skeletons
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                # Logic: One hand vs Two hands
                if num_hands == 1:
                    features = self._get_73_features(results.multi_hand_landmarks[0])
                    hand_type = "single"
                elif num_hands == 2:
                    f1 = self._get_73_features(results.multi_hand_landmarks[0])
                    f2 = self._get_73_features(results.multi_hand_landmarks[1])
                    features = np.concatenate([f1, f2])
                    hand_type = "dual"
                
                # Data Recording
                if self.is_collecting and self.collected_count < self.target_samples:
                    row = list(features) + [self.gesture_label, self.lang_code, hand_type]
                    self.data_rows.append(row)
                    self.collected_count += 1
                    if self.collected_count >= self.target_samples:
                        self.is_collecting = False
                        print(f"✅ Batch complete ({self.target_samples} samples). Press Q to save.")

            # UI Text
            color = (0, 255, 0) if self.is_collecting else (0, 0, 255)
            status = "RECORDING" if self.is_collecting else "PAUSED"
            cv2.putText(frame, f"LANG: {self.lang_code} | GESTURE: {self.gesture_label}", (20, 40), 1, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, f"STATUS: {status}", (20, 80), 1, 1.5, color, 2)
            cv2.putText(frame, f"PROGRESS: {self.collected_count}/{self.target_samples}", (20, 120), 1, 1.5, (255, 255, 0), 2)
            
            if hand_type == "dual" and num_hands < 2:
                 cv2.putText(frame, "NEED BOTH HANDS", (200, 400), 1, 3, (0, 0, 255), 4)

            cv2.imshow("GestureX Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): self.is_collecting = not self.is_collecting
            elif key == ord('r'): 
                self.data_rows = []
                self.collected_count = 0
            elif key == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()

    def save_data(self):
        if not self.data_rows:
            print("No data collected.")
            return
            
        # Determine columns based on max feature count in data_rows
        max_f = len(self.data_rows[0]) - 3
        cols = [f"f{i}" for i in range(max_f)] + ["label", "language", "hand_type"]
        
        df = pd.DataFrame(self.data_rows, columns=cols)
        
        # Append to existing
        if os.path.exists(self.csv_path):
            existing = pd.read_csv(self.csv_path)
            df = pd.concat([existing, df], ignore_index=True)
            
        df.to_csv(self.csv_path, index=False)
        print(f"\n✨ SAVED: {self.csv_path}")
        print(f"Total rows in file: {len(df)}")
        print("\nCounts per Label:")
        print(df['label'].value_counts())
        
        # Critical reminder
        low_labels = df['label'].value_counts()[df['label'].value_counts() < 200].index.tolist()
        if low_labels:
            print(f"⚠️  STILL NEED DATA FOR: {', '.join(low_labels)}")

if __name__ == "__main__":
    collector = GestureCollector()
    collector.start()
