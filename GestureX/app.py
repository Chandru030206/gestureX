import cv2
import mediapipe as mp
import numpy as np
from inference_engine import GestureInferenceEngine
from datetime import datetime

# --- CONFIG ---
LANG_KEYS = {
    ord('1'): 'ASL', ord('2'): 'BSL', ord('3'): 'ISL', ord('4'): 'JSL', 
    ord('5'): 'AUSLAN', ord('6'): 'LSF', ord('7'): 'DGS', ord('8'): 'LIBRAS', 
    ord('9'): 'KSL', ord('0'): 'CSL'
}
# --------------

def run_demo():
    print("🚀 Initializing GestureX Duo Demo...")
    engine = GestureInferenceEngine()
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    sentence = []
    last_committed = None
    flash_text = ""
    flash_timer = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # 1. Inference
        current_gesture, confidence = engine.predict(results)
        
        # 2. Draw Skeletons
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        # 3. Sentence Building
        if current_gesture and current_gesture != last_committed and "HANDS" not in current_gesture:
            sentence.append(current_gesture)
            last_committed = current_gesture

        # --- UI DRAWING ---
        
        # Current Language (Top Left)
        cv2.rectangle(frame, (0,0), (220, 50), (43, 43, 43), -1)
        cv2.putText(frame, f"MODE: {engine.active_lang}", (10, 35), 1, 1.8, (255, 165, 0), 2)
        
        # Prediction Bar (Bottom)
        cv2.rectangle(frame, (0, h-80), (w, h), (30, 30, 30), -1)
        if current_gesture:
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            if "HANDS" in current_gesture: color = (0, 0, 255)
            
            cv2.putText(frame, current_gesture, (w//2 - 100, h-30), 1, 3, color, 3)
            # Confidence bar
            bar_w = int(w * confidence)
            cv2.rectangle(frame, (0, h-5), (bar_w, h), color, -1)

        # Sentence Display (Top Right / Middle)
        sentence_str = " ".join(sentence)
        cv2.putText(frame, f"SENSE: {sentence_str}", (240, 35), 1, 1.5, (255, 255, 255), 2)
        
        # Flash Language Switch
        if flash_timer > 0:
            cv2.putText(frame, flash_text, (w//2 - 150, h//2), 1, 4, (255, 255, 255), 5)
            flash_timer -= 1
            
        # Legend
        cv2.putText(frame, "1-0: LANG | BS: DEL | ENT: CLR | S: SAVE | Q: QUIT", (w-450, h-100), 1, 1, (150, 150, 150), 1)

        cv2.imshow("GestureX Duo Real-Time", frame)
        
        # 4. Keyboard Handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): break
        elif key == 8: # Backspace
            if sentence: sentence.pop()
        elif key == 13: # Enter
            sentence = []
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("output_sentences.txt", "a") as f:
                f.write(f"[{timestamp}] {engine.active_lang}: {sentence_str}\n")
            flash_text = "SENTENCE SAVED"
            flash_timer = 30
        elif key in LANG_KEYS:
            target_lang = LANG_KEYS[key]
            if engine.set_language(target_lang):
                flash_text = f"LOADED {target_lang}"
                flash_timer = 30
                last_committed = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
