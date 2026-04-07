"""
ONNX Landmark Inference Engine
===============================
Uses MediaPipe Hands to extract landmarks, normalizes them relative to the wrist,
and runs inference through ONNX Runtime for word and alphabet models.

Pipeline:
  1. MediaPipe Hands → extract 21 or 42 landmarks (x,y,z)
  2. Normalize landmarks relative to wrist (index 0)
  3. Run word_model.onnx → if confidence > 0.75, return word
  4. Else run alphabet_model.onnx → if confidence > 0.75, return letter
  5. Apply 5-frame temporal smoothing
  6. If no hand for 2s → "No hand detected"
  7. If below threshold → "Uncertain"
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Try importing onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNX = True
    logger.info("✓ ONNX Runtime available")
except ImportError:
    HAS_ONNX = False
    logger.warning("✗ ONNX Runtime not installed — using fallback classifier")

# MediaPipe
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("✗ MediaPipe not installed")


# ═══════════════════════════════════════
# Language Configuration
# ═══════════════════════════════════════

LANGUAGE_CONFIG = {
    "ASL":    {"hands": 1, "features": 63,  "name": "American Sign Language"},
    "BSL":    {"hands": 1, "features": 63,  "name": "British Sign Language"},
    "ISL":    {"hands": 1, "features": 63,  "name": "Indian Sign Language"},
    "KSL":    {"hands": 2, "features": 126, "name": "Korean Sign Language"},
    "JSL":    {"hands": 1, "features": 63,  "name": "Japanese Sign Language"},
    "CSL":    {"hands": 2, "features": 126, "name": "Chinese Sign Language"},
    "AUSLAN": {"hands": 1, "features": 63,  "name": "Australian Sign Language"},
    "LSF":    {"hands": 1, "features": 63,  "name": "French Sign Language"},
    "DGS":    {"hands": 2, "features": 126, "name": "German Sign Language"},
    "RSL":    {"hands": 1, "features": 63,  "name": "Russian Sign Language"},
}

# Default word labels (used when no labels file exists)
DEFAULT_WORD_LABELS = [
    "HELLO", "YES", "NO", "THANK YOU", "PLEASE", "SORRY",
    "HELP", "STOP", "GOODBYE", "I LOVE YOU", "BLANK"
]

DEFAULT_ALPHABET_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["BLANK"]


class LandmarkExtractor:
    """Extracts hand landmarks using MediaPipe Hands."""

    def __init__(self, max_num_hands: int = 2):
        if not HAS_MEDIAPIPE:
            raise RuntimeError("MediaPipe is required for landmark extraction")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        )

    def extract(self, frame: np.ndarray, num_hands_required: int = 1) -> Optional[np.ndarray]:
        """
        Extract normalized landmarks from frame.
        
        Returns:
            numpy array of shape (features,) or None if no hand detected.
            Single hand: 63 features (21 landmarks × 3)
            Two hands: 126 features (42 landmarks × 3)
        """
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hands = results.multi_hand_landmarks
        
        if num_hands_required == 1:
            # Use first detected hand
            hand = hands[0]
            landmarks = self._hand_to_array(hand)
            return self._normalize_wrist_relative(landmarks)
        else:
            # Need 2 hands
            if len(hands) < 2:
                # Pad with zeros if only 1 hand detected
                hand1 = self._hand_to_array(hands[0])
                hand1_norm = self._normalize_wrist_relative(hand1)
                hand2_norm = np.zeros(63, dtype=np.float32)
                return np.concatenate([hand1_norm, hand2_norm])
            else:
                hand1 = self._hand_to_array(hands[0])
                hand2 = self._hand_to_array(hands[1])
                h1_norm = self._normalize_wrist_relative(hand1)
                h2_norm = self._normalize_wrist_relative(hand2)
                return np.concatenate([h1_norm, h2_norm])

    def _hand_to_array(self, hand_landmarks) -> np.ndarray:
        """Convert hand landmarks to flat array [x0,y0,z0, x1,y1,z1, ...]"""
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def _normalize_wrist_relative(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist (index 0).
        Subtract wrist position from all landmarks.
        """
        # Reshape to (21, 3)
        pts = landmarks.reshape(-1, 3)
        wrist = pts[0].copy()
        
        # Subtract wrist position
        pts = pts - wrist
        
        return pts.flatten().astype(np.float32)


class ModelPredictor:
    """Generic predictor for ONNX or TFLite models."""

    def __init__(self, model_path: str, labels: List[str]):
        self.labels = labels
        self.model_path = Path(model_path)
        self.is_tflite = self.model_path.suffix == ".tflite"

        if self.is_tflite:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"TFLite model loaded: {model_path} ({len(labels)} classes)")
        else:
            if not HAS_ONNX:
                raise RuntimeError("ONNX Runtime required")
            self.session = ort.InferenceSession(str(model_path))
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"ONNX model loaded: {model_path} ({len(labels)} classes)")

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Run prediction. Returns: (label, confidence)
        """
        x = features.reshape(1, -1).astype(np.float32)

        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], x)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            outputs = self.session.run(None, {self.input_name: x})
            probs = outputs[0][0]

        # Softmax fallback
        if np.any(probs < 0) or np.sum(probs) < 0.5:
            probs = self._softmax(probs)

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else "UNKNOWN"

        return label, confidence

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()


class FallbackClassifier:
    """Robust rule-based classifier using landmark geometry."""

    def __init__(self, language: str):
        self.language = language

    def predict_word(self, features: np.ndarray) -> Tuple[str, float]:
        """Heuristic-based ASL word detection."""
        if features is None or len(features) < 63:
            return "NONE", 0.0

        pts = features.reshape(-1, 3)
        
        # Heuristics: Extended finger detection (tip y < base y)
        # Note: MediaPipe Y increases downwards. Tip y < Pip y means finger is extended UP.
        is_thumb_up = pts[4][0] > pts[3][0] + 0.05
        is_index_up = pts[8][1] < pts[6][1] - 0.05
        is_middle_up = pts[12][1] < pts[10][1] - 0.05
        is_ring_up = pts[16][1] < pts[14][1] - 0.05
        is_pinky_up = pts[20][1] < pts[18][1] - 0.05

        extended_count = sum([is_index_up, is_middle_up, is_ring_up, is_pinky_up])

        # HELLO: Open palm (all up/spread)
        if extended_count >= 4 and is_thumb_up:
            return "HELLO", 0.92
        # YES: Fist (0 up)
        elif extended_count == 0:
            return "YES", 0.88
        # NO: Index/Middle up
        elif extended_count == 2 and is_index_up and is_middle_up:
            return "NO", 0.85
        # I LOVE YOU: Thumb, Index, Pinky
        elif is_thumb_up and is_index_up and is_pinky_up and not is_middle_up:
            return "I LOVE YOU", 0.90
        # THANK YOU: Flat hand forward (leaning y)
        elif extended_count >= 4:
            return "THANK YOU", 0.80

        return "NONE", 0.50

    def predict_letter(self, features: np.ndarray) -> Tuple[str, float]:
        """Heuristic-based ASL letter detection."""
        if features is None or len(features) < 63:
            return "NONE", 0.0
            
        pts = features.reshape(-1, 3)
        
        is_thumb_up = pts[4][0] > pts[3][0] + 0.05
        is_index_up = pts[8][1] < pts[6][1] - 0.05
        is_middle_up = pts[12][1] < pts[10][1] - 0.05
        is_ring_up = pts[16][1] < pts[14][1] - 0.05
        is_pinky_up = pts[20][1] < pts[18][1] - 0.05
        extended_count = sum([is_index_up, is_middle_up, is_ring_up, is_pinky_up])

        # A: Fist (0 up)
        if extended_count == 0: return "A", 0.85
        # B: 4 Fingers up
        if extended_count == 4 and not is_thumb_up: return "B", 0.88
        # D: Index up
        if extended_count == 1 and is_index_up: return "D", 0.90
        # C: Curved
        if 0 < extended_count < 4: return "C", 0.75
        
        return "NONE", 0.40
        if len(pts) >= 21:
            thumb_tip = pts[4]
            index_tip = pts[8]
            
            dist = np.linalg.norm(thumb_tip - index_tip)
            
            if dist < 0.15:
                return "O", 0.70
            elif dist > 0.5:
                return "Y", 0.65
            else:
                return "A", 0.60
        
        return "BLANK", 0.50


class GestureRecognizerONNX:
    """
    Complete gesture recognition pipeline.
    
    For each language:
      1. Load word_model.onnx and alphabet_model.onnx
      2. Extract landmarks via MediaPipe
      3. Run word model first, fallback to alphabet
      4. Apply 5-frame smoothing
    """

    def __init__(self, language: str = "ASL", models_dir: str = None):
        self.language = language.upper()
        self.config = LANGUAGE_CONFIG.get(self.language, LANGUAGE_CONFIG["ASL"])
        self.num_hands = self.config["hands"]
        self.num_features = self.config["features"]
        
        # Models directory
        if models_dir is None:
            models_dir = str(Path(__file__).parent.parent / "models")
        
        self.models_dir = Path(models_dir)
        
        # Initialize landmark extractor
        try:
            self.extractor = LandmarkExtractor(max_num_hands=2)
        except Exception as e:
            logger.warning(f"LandmarkExtractor init failed: {e}")
            self.extractor = None
        
        # Load ONNX models
        self.word_predictor = None
        self.alphabet_predictor = None
        self.fallback = FallbackClassifier(self.language)
        
        self._load_models()
        
        # Temporal smoothing (5-frame buffer)
        self.prediction_buffer = deque(maxlen=5)
        self.last_hand_detected = time.time()
        
        # Confidence threshold
        self.confidence_threshold = 0.75

    def _load_models(self):
        """Load ONNX/TFLite models for the current language."""
        lang_dir = self.models_dir / self.language
        
        # 1. Alphabet model (A-Z)
        alpha_base = lang_dir / "alphabet_model"
        alpha_labels_path = lang_dir / "alphabet_labels.txt"
        
        for ext in [".tflite", ".onnx"]:
            path = alpha_base.with_suffix(ext)
            if path.exists():
                labels = self._load_labels(alpha_labels_path, DEFAULT_ALPHABET_LABELS)
                self.alphabet_predictor = ModelPredictor(str(path), labels)
                break
        
        # 2. Word model (HELLO, etc)
        word_base = lang_dir / "word_model"
        word_labels_path = lang_dir / "word_labels.txt"
        
        for ext in [".tflite", ".onnx"]:
            path = word_base.with_suffix(ext)
            if path.exists():
                labels = self._load_labels(word_labels_path, DEFAULT_WORD_LABELS)
                self.word_predictor = ModelPredictor(str(path), labels)
                break

    def _load_labels(self, path: Path, default: List[str]) -> List[str]:
        """Load labels from text file, one per line."""
        if path.exists():
            with open(path, 'r') as f:
                labels = [line.strip() for line in f if line.strip()]
            if labels:
                return labels
        return default

    def recognize(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Full recognition pipeline.
        
        Returns:
            {
                "success": bool,
                "gesture": str,
                "text": str,
                "confidence": float,
                "mode": "word" | "alphabet" | "no_hand",
                "language": str,
            }
        """
        # Extract landmarks
        features = None
        if self.extractor:
            try:
                features = self.extractor.extract(frame, self.num_hands)
            except Exception as e:
                logger.debug(f"Landmark extraction error: {e}")
        
        if features is None:
            # No hand detected
            elapsed = time.time() - self.last_hand_detected
            if elapsed > 2.0:
                self.prediction_buffer.clear()
                return {
                    "success": True,
                    "gesture": "BLANK",
                    "text": "No hand detected",
                    "confidence": 0.0,
                    "mode": "no_hand",
                    "language": self.language,
                }
            else:
                # Grace period — return last known prediction
                return {
                    "success": True,
                    "gesture": "BLANK",
                    "text": "BLANK",
                    "confidence": 0.95,
                    "mode": "no_hand",
                    "language": self.language,
                }
        
        self.last_hand_detected = time.time()
        
        # Verify feature count matches expected
        if len(features) != self.num_features:
            # Pad or truncate
            if len(features) < self.num_features:
                features = np.pad(features, (0, self.num_features - len(features)))
            else:
                features = features[:self.num_features]
        
        # 1. Try word model first
        word_label, word_conf = self._predict_word(features)
        
        if word_conf >= self.confidence_threshold and not self._is_blank(word_label):
            # Smooth with buffer
            self.prediction_buffer.append((word_label, word_conf))
            smoothed = self._get_smoothed_prediction()
            
            return {
                "success": True,
                "gesture": smoothed[0],
                "text": smoothed[0],
                "confidence": smoothed[1],
                "mode": "word",
                "language": self.language,
            }
        
        # 2. Try alphabet model
        letter_label, letter_conf = self._predict_letter(features)
        
        if letter_conf >= self.confidence_threshold and not self._is_blank(letter_label):
            self.prediction_buffer.append((letter_label, letter_conf))
            smoothed = self._get_smoothed_prediction()
            
            return {
                "success": True,
                "gesture": smoothed[0],
                "text": smoothed[0],
                "confidence": smoothed[1],
                "mode": "alphabet",
                "language": self.language,
            }
        
        # Below threshold
        best_label = word_label if word_conf >= letter_conf else letter_label
        best_conf = max(word_conf, letter_conf)
        
        return {
            "success": True,
            "gesture": best_label,
            "text": "Uncertain",
            "confidence": best_conf,
            "mode": "uncertain",
            "language": self.language,
        }

    def _predict_word(self, features: np.ndarray) -> Tuple[str, float]:
        """Run word prediction."""
        if self.word_predictor:
            return self.word_predictor.predict(features)
        return self.fallback.predict_word(features)

    def _predict_letter(self, features: np.ndarray) -> Tuple[str, float]:
        """Run alphabet prediction."""
        if self.alphabet_predictor:
            return self.alphabet_predictor.predict(features)
        return self.fallback.predict_letter(features)

    def _is_blank(self, label: str) -> bool:
        return label.upper().strip() in ("BLANK", "")

    def _get_smoothed_prediction(self) -> Tuple[str, float]:
        """
        Apply 5-frame smoothing: return the most frequent prediction 
        in the buffer with averaged confidence.
        """
        if not self.prediction_buffer:
            return ("BLANK", 0.0)
        
        # Count occurrences
        counts = {}
        for label, conf in self.prediction_buffer:
            if label not in counts:
                counts[label] = {"count": 0, "total_conf": 0.0}
            counts[label]["count"] += 1
            counts[label]["total_conf"] += conf
        
        # Find most frequent
        best = max(counts.items(), key=lambda x: x[1]["count"])
        label = best[0]
        avg_conf = best[1]["total_conf"] / best[1]["count"]
        
        return (label, avg_conf)
