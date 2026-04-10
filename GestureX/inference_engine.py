import onnxruntime as ort
import numpy as np
import joblib
import os
import json
from collections import Counter, deque

# --- CONFIG ---
MODELS_ROOT = "models"
# --------------

class TemporalSmoother:
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)
    def add(self, prediction): self.buffer.append(prediction)
    def reset(self): self.buffer.clear()
    def get_voted_result(self, majority=3):
        if len(self.buffer) < self.buffer.maxlen: return None
        counts = Counter(self.buffer)
        winner, count = counts.most_common(1)[0]
        return winner if count >= majority else None

class GestureInferenceEngine:
    def __init__(self):
        self.models = {}
        self.active_lang = "ASL"
        self.smoother = TemporalSmoother(size=5)
        self.load_all_models()

    def load_all_models(self):
        if not os.path.exists(MODELS_ROOT):
            print(f"❌ Error: {MODELS_ROOT} folder not found.")
            return

        print("\n" + "="*40)
        print("🧠 LOADING MULTI-LANGUAGE MODELS")
        print("="*40)

        for lang in os.listdir(MODELS_ROOT):
            lang_dir = os.path.join(MODELS_ROOT, lang)
            if not os.path.isdir(lang_dir): continue
            
            info_path = os.path.join(lang_dir, "model_info.json")
            if not os.path.exists(info_path): continue
            
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                # Check for single vs dual model
                model_name = "word_model.onnx" if info['hand_type'] == "single" else "word_model_dual.onnx"
                model_path = os.path.join(lang_dir, model_name)
                
                if not os.path.exists(model_path): continue
                
                self.models[lang] = {
                    "session": ort.InferenceSession(model_path),
                    "scaler": joblib.load(os.path.join(lang_dir, "scaler.pkl")),
                    "encoder": joblib.load(os.path.join(lang_dir, "label_encoder.pkl")),
                    "info": info
                }
                print(f"✅ Loaded {lang} ({info['hand_type']} hand, {len(info['classes'])} gestures)")
            except Exception as e:
                print(f"❌ Error loading {lang}: {e}")

        if not self.models:
            print("⚠️ No valid models found. Please run train_language_model.py first.")
        else:
            first_lang = list(self.models.keys())[0]
            self.set_language(first_lang if "ASL" not in self.models else "ASL")

    def set_language(self, lang_code):
        if lang_code in self.models:
            self.active_lang = lang_code
            self.smoother.reset()
            info = self.models[lang_code]['info']
            print(f"🌐 Switched to {lang_code} | Mode: {info['hand_type']} | Gestures: {len(info['classes'])}")
            return True
        return False

    def _get_73_features(self, hand_lms):
        """Standard 73-feature normalization."""
        lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist = lms[0].copy()
        lms = lms - wrist
        dist = np.linalg.norm(lms[0] - lms[9])
        if dist > 0: lms = lms / dist
        flat = lms.flatten()
        tips = [4, 8, 12, 16, 20]
        distances = [np.linalg.norm(lms[tips[i]] - lms[tips[j]]) 
                      for i in range(len(tips)) for j in range(i+1, len(tips))]
        return np.concatenate([flat, distances])

    def predict(self, results):
        """
        Main inference call.
        results: MediaPipe results object
        """
        if self.active_lang not in self.models: return None, 0.0
        
        m_data = self.models[self.active_lang]
        info = m_data['info']
        
        if not results.multi_hand_landmarks:
            self.smoother.reset()
            return None, 0.0

        # Construct features based on expected input size
        num_found = len(results.multi_hand_landmarks)
        features = None
        
        if info['hand_type'] == "single":
            # Just take the first detected hand
            features = self._get_73_features(results.multi_hand_landmarks[0])
        elif info['hand_type'] == "dual":
            if num_found < 2:
                # Warning for 2-handed languages
                return "NEED BOTH HANDS", 0.0
            f1 = self._get_73_features(results.multi_hand_landmarks[0])
            f2 = self._get_73_features(results.multi_hand_landmarks[1])
            features = np.concatenate([f1, f2])

        # Scale and Predict
        X_scaled = m_data['scaler'].transform(features.reshape(1, -1)).astype(np.float32)
        sess = m_data['session']
        outputs = sess.run(None, {sess.get_inputs()[0].name: X_scaled})
        
        probs_dict = outputs[1][0] if isinstance(outputs[1], list) else outputs[1]
        top_idx = np.argmax(list(probs_dict.values()))
        top_conf = max(probs_dict.values())
        top_label = m_data['encoder'].inverse_transform([top_idx])[0]

        if top_conf < 0.60:
            self.smoother.add(None)
            return None, top_conf
            
        self.smoother.add(top_label)
        final_label = self.smoother.get_voted_result()
        
        return final_label, top_conf

if __name__ == "__main__":
    print("Testing Multi-Language Engine Registry...")
    engine = GestureInferenceEngine()
