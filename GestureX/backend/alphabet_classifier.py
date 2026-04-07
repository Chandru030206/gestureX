"""
Alphabet Classifier Module for Fingerspelling Detection
========================================================

Loads language-specific pretrained alphabet models and classifies
hand landmarks into alphabet letters (A-Z + BLANK/PAUSE).

Each sign language has its OWN alphabet handshapes - models are separate per language.

Supported Languages:
- ASL (American Sign Language)
- BSL (British Sign Language)  
- ISL (Indian Sign Language)
- JSL (Japanese Sign Language)
- Extensible to more languages

Model files expected at:
    pretrained_models/<LANG>_alphabet.onnx
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import ONNX runtime for model inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available - using fallback classifier")

# Alphabet labels (A-Z + special tokens)
ALPHABET_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["BLANK", "PAUSE", "SPACE"]

# Language-specific alphabet info
LANGUAGE_ALPHABETS = {
    "ASL": {
        "name": "American Sign Language",
        "alphabet_count": 26,
        "has_motion_letters": ["J", "Z"],  # Letters requiring motion in ASL
        "notes": "One-handed alphabet, J and Z require motion"
    },
    "BSL": {
        "name": "British Sign Language",
        "alphabet_count": 26,
        "two_handed": True,  # BSL uses two-handed alphabet
        "notes": "Two-handed alphabet system"
    },
    "ISL": {
        "name": "Indian Sign Language",
        "alphabet_count": 26,
        "notes": "Based on ASL with regional variations"
    },
    "JSL": {
        "name": "Japanese Sign Language",
        "alphabet_count": 26,  # For English letters
        "notes": "Has both Japanese kana and English alphabet signs"
    },
    "AUSLAN": {
        "name": "Australian Sign Language",
        "alphabet_count": 26,
        "two_handed": True,
        "notes": "Similar to BSL, two-handed system"
    },
    "LSF": {
        "name": "French Sign Language",
        "alphabet_count": 26,
        "notes": "One-handed alphabet"
    },
    "DGS": {
        "name": "German Sign Language",
        "alphabet_count": 26,
        "notes": "One-handed alphabet with umlauts"
    }
}


@dataclass
class AlphabetPrediction:
    """Result of alphabet classification"""
    letter: str
    confidence: float
    language: str
    is_blank: bool = False
    raw_scores: Optional[Dict[str, float]] = None


class FallbackAlphabetClassifier:
    """
    Fallback classifier when ONNX models are not available.
    Uses heuristic-based classification from hand landmarks.
    
    This provides basic functionality for demonstration but
    real deployment should use trained ONNX models.
    """
    
    def __init__(self, language: str):
        self.language = language
        self.labels = ALPHABET_LABELS
        logger.info(f"Initialized fallback classifier for {language}")
    
    def classify(self, landmarks: np.ndarray) -> AlphabetPrediction:
        """
        Classify hand landmarks into an alphabet letter.
        
        Args:
            landmarks: Array of shape (21, 3) for 21 hand landmarks with x, y, z
        
        Returns:
            AlphabetPrediction with detected letter and confidence
        """
        if landmarks is None or len(landmarks) == 0:
            return AlphabetPrediction(
                letter="BLANK",
                confidence=1.0,
                language=self.language,
                is_blank=True
            )
        
        # Normalize landmarks
        landmarks = np.array(landmarks).reshape(21, 3)
        
        # Extract key features for heuristic classification
        # Fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        # MCP (base) indices: thumb=2, index=5, middle=9, ring=13, pinky=17
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        wrist = landmarks[0]
        
        # Calculate finger extension (tip higher than MCP = extended)
        def is_extended(tip, mcp, threshold=0.03):
            return tip[1] < mcp[1] - threshold  # y decreases upward
        
        thumb_extended = thumb_tip[0] < thumb_mcp[0] - 0.05  # thumb extends sideways
        index_extended = is_extended(index_tip, index_mcp)
        middle_extended = is_extended(middle_tip, middle_mcp)
        ring_extended = is_extended(ring_tip, ring_mcp)
        pinky_extended = is_extended(pinky_tip, pinky_mcp)
        
        extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
        
        # Simple heuristic rules for common letters
        letter = "BLANK"
        confidence = 0.5
        
        # A: Fist with thumb to side
        if extended_count == 0 and thumb_extended:
            letter = "A"
            confidence = 0.7
        
        # B: All fingers up, thumb across palm
        elif extended_count == 4 and not thumb_extended:
            letter = "B"
            confidence = 0.7
        
        # C: Curved hand like holding a cup
        elif extended_count >= 3:
            # Check if fingers are curved (tips closer together than MCPs)
            tip_spread = np.linalg.norm(index_tip - pinky_tip)
            mcp_spread = np.linalg.norm(index_mcp - pinky_mcp)
            if tip_spread < mcp_spread * 0.8:
                letter = "C"
                confidence = 0.65
        
        # D: Index up, others closed, thumb touches middle
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            letter = "D"
            confidence = 0.7
        
        # E: All fingers curled, thumb across
        elif extended_count == 0 and not thumb_extended:
            letter = "E"
            confidence = 0.6
        
        # F: Index and thumb form circle, others up
        elif middle_extended and ring_extended and pinky_extended and not index_extended:
            dist_thumb_index = np.linalg.norm(thumb_tip - index_tip)
            if dist_thumb_index < 0.05:
                letter = "F"
                confidence = 0.65
        
        # I: Pinky only extended
        elif pinky_extended and not index_extended and not middle_extended and not ring_extended:
            letter = "I"
            confidence = 0.7
        
        # L: Index and thumb form L shape
        elif index_extended and thumb_extended and not middle_extended and not ring_extended and not pinky_extended:
            letter = "L"
            confidence = 0.7
        
        # O: All fingers curved to form O with thumb
        elif extended_count == 0:
            dist_thumb_index = np.linalg.norm(thumb_tip - index_tip)
            if dist_thumb_index < 0.06:
                letter = "O"
                confidence = 0.6
        
        # V: Index and middle extended (peace sign)
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            letter = "V"
            confidence = 0.75
        
        # W: Index, middle, ring extended
        elif index_extended and middle_extended and ring_extended and not pinky_extended:
            letter = "W"
            confidence = 0.7
        
        # Y: Thumb and pinky extended (hang loose)
        elif thumb_extended and pinky_extended and not index_extended and not middle_extended and not ring_extended:
            letter = "Y"
            confidence = 0.75
        
        # Default: if no clear match, return most likely based on finger count
        if letter == "BLANK" and extended_count > 0:
            # Map extended finger count to likely letters
            finger_map = {
                1: "D",  # One finger = D or similar
                2: "V",  # Two fingers = V
                3: "W",  # Three fingers = W
                4: "B",  # Four fingers = B
                5: "B"   # All five (with thumb) = open hand
            }
            if thumb_extended:
                extended_count += 1
            letter = finger_map.get(extended_count, "BLANK")
            confidence = 0.4
        
        return AlphabetPrediction(
            letter=letter,
            confidence=confidence,
            language=self.language,
            is_blank=(letter == "BLANK")
        )


class ONNXAlphabetClassifier:
    """
    ONNX-based alphabet classifier for production use.
    Loads pretrained models for each sign language.
    """
    
    def __init__(self, language: str, model_path: str):
        self.language = language
        self.model_path = model_path
        self.labels = ALPHABET_LABELS
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        logger.info(f"Loaded ONNX model for {language}: {model_path}")
    
    def classify(self, landmarks: np.ndarray) -> AlphabetPrediction:
        """
        Classify hand landmarks using ONNX model.
        """
        if landmarks is None or len(landmarks) == 0:
            return AlphabetPrediction(
                letter="BLANK",
                confidence=1.0,
                language=self.language,
                is_blank=True
            )
        
        # Prepare input
        landmarks = np.array(landmarks, dtype=np.float32).flatten()
        
        # Pad or truncate to expected input size
        expected_size = 63  # 21 landmarks * 3 coordinates
        if len(landmarks) < expected_size:
            landmarks = np.pad(landmarks, (0, expected_size - len(landmarks)))
        elif len(landmarks) > expected_size:
            landmarks = landmarks[:expected_size]
        
        # Reshape for model input
        input_data = landmarks.reshape(1, -1).astype(np.float32)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        scores = outputs[0][0]
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        # Get top prediction
        top_idx = np.argmax(probs)
        top_letter = self.labels[top_idx] if top_idx < len(self.labels) else "BLANK"
        top_confidence = float(probs[top_idx])
        
        # Build raw scores dict
        raw_scores = {self.labels[i]: float(probs[i]) for i in range(min(len(probs), len(self.labels)))}
        
        return AlphabetPrediction(
            letter=top_letter,
            confidence=top_confidence,
            language=self.language,
            is_blank=(top_letter in ["BLANK", "PAUSE"]),
            raw_scores=raw_scores
        )


class AlphabetClassifierManager:
    """
    Manages alphabet classifiers for multiple sign languages.
    Loads appropriate model based on selected language.
    """
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "pretrained_models")
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.classifiers: Dict[str, object] = {}
        self.supported_languages = list(LANGUAGE_ALPHABETS.keys())
        
        logger.info(f"AlphabetClassifierManager initialized. Models dir: {self.models_dir}")
    
    def get_classifier(self, language: str):
        """
        Get or create classifier for the specified language.
        Uses ONNX model if available, otherwise falls back to heuristic classifier.
        """
        language = language.upper()
        
        if language not in self.supported_languages:
            logger.warning(f"Language {language} not supported, defaulting to ASL")
            language = "ASL"
        
        # Return cached classifier if available
        if language in self.classifiers:
            return self.classifiers[language]
        
        # Try to load ONNX model
        model_path = self.models_dir / f"{language}_alphabet.onnx"
        
        if ONNX_AVAILABLE and model_path.exists():
            classifier = ONNXAlphabetClassifier(language, str(model_path))
        else:
            logger.info(f"No ONNX model found for {language}, using fallback classifier")
            classifier = FallbackAlphabetClassifier(language)
        
        self.classifiers[language] = classifier
        return classifier
    
    def classify(self, language: str, landmarks: np.ndarray) -> AlphabetPrediction:
        """
        Classify landmarks for the specified language.
        """
        classifier = self.get_classifier(language)
        return classifier.classify(landmarks)
    
    def get_language_info(self, language: str) -> dict:
        """
        Get information about a language's alphabet system.
        """
        language = language.upper()
        if language in LANGUAGE_ALPHABETS:
            return LANGUAGE_ALPHABETS[language]
        return None
    
    def list_available_models(self) -> List[str]:
        """
        List languages with available ONNX models.
        """
        available = []
        for lang in self.supported_languages:
            model_path = self.models_dir / f"{lang}_alphabet.onnx"
            if model_path.exists():
                available.append(lang)
        return available


# Global instance
_classifier_manager: Optional[AlphabetClassifierManager] = None


def get_classifier_manager() -> AlphabetClassifierManager:
    """Get the global classifier manager instance."""
    global _classifier_manager
    if _classifier_manager is None:
        _classifier_manager = AlphabetClassifierManager()
    return _classifier_manager
