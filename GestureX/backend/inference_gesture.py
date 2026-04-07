"""
Gesture Recognition Engine - GestureX Duo
=========================================
Implements landmark-based ONNX inference with MediaPipe preprocessing.
Supports 10 isolated languages with Word + Alphabet dual model structure.
"""

import os
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import onnxruntime as ort
import mediapipe as mp

logger = logging.getLogger(__name__)

# ─── Configuration ───
LANGUAGE_HANDS = {
    "ASL": 1, "BSL": 1, "ISL": 1, "KSL": 2, "JSL": 1,
    "CSL": 2, "AUSLAN": 1, "LSF": 1, "DGS": 2, "RSL": 1
}

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

class GestureRecognizer:
    """GestureX Duo Recognition Pipeline."""
    
    def __init__(self, language: str = "ASL"):
        self.language = language.upper()
        self.num_hands = LANGUAGE_HANDS.get(self.language, 1)
        self.num_features = self.num_hands * 63 # 21 * 3 per hand
        
        # Inference sessions
        self.word_session = None
        self.alpha_session = None
        self.word_labels = []
        self.alpha_labels = []
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Smoothing state
        self.history = []
        self.max_history = 5
        self.last_detection_time = time.time()
        
        self._load_models()

    def _load_models(self):
        """Load ONNX models and labels for the selected language."""
        lang_dir = MODELS_DIR / self.language
        
        try:
            word_path = lang_dir / "word_model.onnx"
            alpha_path = lang_dir / "alphabet_model.onnx"
            word_label_path = lang_dir / "word_labels.txt"
            alpha_label_path = lang_dir / "alphabet_labels.txt"
            
            if word_path.exists():
                self.word_session = ort.InferenceSession(str(word_path))
                logger.info(f"✓ Word model loaded for {self.language}")
            
            if alpha_path.exists():
                self.alpha_session = ort.InferenceSession(str(alpha_path))
                logger.info(f"✓ Alphabet model loaded for {self.language}")
            
            if word_label_path.exists():
                with open(word_label_path, 'r') as f:
                    self.word_labels = [l.strip() for l in f.readlines()]
            
            if alpha_label_path.exists():
                with open(alpha_label_path, 'r') as f:
                    self.alpha_labels = [l.strip() for l in f.readlines()]
                    
        except Exception as e:
            logger.error(f"Error loading models for {self.language}: {e}")

    def normalize_landmarks(self, multi_hand_landmarks) -> Optional[np.ndarray]:
        """Normalize landmarks relative to wrist (index 0)."""
        all_features = []
        
        # Sort hands by X to ensure consistent order for 2-hand languages
        sorted_hands = sorted(multi_hand_landmarks, key=lambda x: x.landmark[0].x)
        
        for i, hand_lms in enumerate(sorted_hands):
            if i >= self.num_hands:
                break
            
            # Extract x, y, z
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            
            # Wrist (0) as origin
            wrist = pts[0].copy()
            pts = pts - wrist
            
            # Normalize scale (consistent with training script)
            max_val = np.max(np.abs(pts))
            if max_val > 0:
                pts = pts / max_val
            
            all_features.extend(pts.flatten().tolist())
        
        # Pad with zeros if less hands than expected
        while len(all_features) < self.num_features:
            all_features.extend([0.0] * 63)
            
        return np.array(all_features[:self.num_features], dtype=np.float32).reshape(1, -1)

    def predict(self, features: np.ndarray, session: ort.InferenceSession, labels: List[str]) -> Tuple[str, float]:
        """Run ONNX inference."""
        if not session or not labels:
            return "Unknown", 0.0
            
        inputs = {session.get_inputs()[0].name: features}
        outputs = session.run(None, inputs)
        probs = outputs[0][0]
        
        idx = np.argmax(probs)
        confidence = float(probs[idx])
        label = labels[idx] if idx < len(labels) else "Unknown"
        
        return label, confidence

    def recognize(self, frame: np.ndarray) -> Dict[str, Any]:
        """Full pipeline: Landmarks → Word Model → (Fallback) Alpha Model → Smoothing."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        now = time.time()
        
        if not results.multi_hand_landmarks:
            if now - self.last_detection_time > 2.0:
                self.history = []
                return {"success": False, "error": "No hand detected", "display_text": "No hand detected"}
            return {"success": False, "error": "Hand lost", "display_text": self._get_smoothed_result()}

        self.last_detection_time = now
        features = self.normalize_landmarks(results.multi_hand_landmarks)
        
        if features is None:
            return {"success": False, "error": "Normalization failed"}

        # 1. Run Word Model
        word, word_conf = self.predict(features, self.word_session, self.word_labels)
        
        final_label = "Uncertain"
        final_conf = 0.0
        result_type = "uncertain"

        if word_conf >= 0.75:
            final_label = word
            final_conf = word_conf
            result_type = "word"
        else:
            # 2. Run Alphabet Model if word failed
            alpha, alpha_conf = self.predict(features, self.alpha_session, self.alpha_labels)
            if alpha_conf >= 0.75:
                final_label = alpha
                final_conf = alpha_conf
                result_type = "alphabet"
            else:
                final_label = "Uncertain"
                final_conf = max(word_conf, alpha_conf)
                result_type = "uncertain"

        # Apply basic 5-frame smoothing
        self.history.append(final_label)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        smoothed_label = self._get_smoothed_result()
        
        return {
            "success": True,
            "gesture": smoothed_label,
            "text": smoothed_label,
            "confidence": final_conf,
            "type": result_type,
            "language": self.language
        }

    def _get_smoothed_result(self) -> str:
        """Return the most frequent label in the history if it meets threshold."""
        if not self.history:
            return "No hand detected"
        
        counts = {}
        for label in self.history:
            counts[label] = counts.get(label, 0) + 1
        
        # Sort by count
        best_label = max(counts, key=counts.get)
        
        # Require majority for smoothing stability
        if counts[best_label] >= 3:
            return best_label
        return "Uncertain"
