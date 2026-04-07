"""
Fingerspelling Detector Module
==============================

Detects alphabet gestures in sequence, combines them into words/names,
and outputs the detected name as text.

Features:
- Sequences individual letter detections into words
- Handles pause detection for word boundaries
- Confidence-based filtering to avoid noise
- Support for multiple sign languages

Usage:
    detector = FingerspellingDetector("ASL")
    detector.add_detection(letter="A", confidence=0.9)
    detector.add_detection(letter="R", confidence=0.85)
    ...
    result = detector.get_result()
    # Returns: {"detected_letters": ["A","R","U","N"], "detected_name": "ARUN", ...}
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class DetectionState(Enum):
    """States for the fingerspelling detection state machine"""
    IDLE = "idle"
    DETECTING = "detecting"
    PAUSED = "paused"
    COMPLETE = "complete"


@dataclass
class LetterDetection:
    """Single letter detection with metadata"""
    letter: str
    confidence: float
    timestamp: float
    frame_count: int = 1


@dataclass
class FingerspellingResult:
    """Result of fingerspelling detection"""
    detected_letters: List[str]
    detected_name: str
    confidence: float
    language: str
    is_complete: bool
    detection_time_ms: float
    letter_confidences: List[float] = field(default_factory=list)


class FingerspellingDetector:
    """
    Detects fingerspelled words by sequencing individual letter detections.
    
    Configuration:
    - min_confidence: Minimum confidence to accept a letter detection
    - min_hold_frames: Minimum frames a letter must be held to be accepted
    - pause_threshold_ms: Time of no detection before considering word complete
    - max_word_length: Maximum letters in a single word
    """
    
    def __init__(
        self,
        language: str = "ASL",
        min_confidence: float = 0.55,
        min_hold_frames: int = 3,
        pause_threshold_ms: float = 1500,
        max_word_length: int = 20
    ):
        self.language = language.upper()
        self.min_confidence = min_confidence
        self.min_hold_frames = min_hold_frames
        self.pause_threshold_ms = pause_threshold_ms
        self.max_word_length = max_word_length
        
        # Detection state
        self.state = DetectionState.IDLE
        self.detected_letters: List[LetterDetection] = []
        self.current_letter: Optional[str] = None
        self.current_letter_frames: int = 0
        self.current_letter_confidence_sum: float = 0.0
        
        # Timing
        self.start_time: Optional[float] = None
        self.last_detection_time: Optional[float] = None
        
        # Recent detections buffer for smoothing
        self.detection_buffer: deque = deque(maxlen=10)
        
        # Statistics
        self.total_frames_processed: int = 0
        self.blank_frames: int = 0
        
        logger.info(f"FingerspellingDetector initialized for {self.language}")
    
    def reset(self):
        """Reset the detector to initial state"""
        self.state = DetectionState.IDLE
        self.detected_letters = []
        self.current_letter = None
        self.current_letter_frames = 0
        self.current_letter_confidence_sum = 0.0
        self.start_time = None
        self.last_detection_time = None
        self.detection_buffer.clear()
        self.total_frames_processed = 0
        self.blank_frames = 0
        logger.debug("Detector reset")
    
    def add_detection(
        self,
        letter: str,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> Optional[FingerspellingResult]:
        """
        Add a single frame detection.
        
        Args:
            letter: Detected letter (A-Z, BLANK, PAUSE, SPACE)
            confidence: Detection confidence (0-1)
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            FingerspellingResult if word is complete, None otherwise
        """
        if timestamp is None:
            timestamp = time.time() * 1000  # ms
        
        self.total_frames_processed += 1
        
        # Start timing on first detection
        if self.start_time is None:
            self.start_time = timestamp
        
        # Add to buffer for smoothing
        self.detection_buffer.append((letter, confidence, timestamp))
        
        # Handle special cases
        if letter in ["BLANK", "PAUSE", "SPACE", ""]:
            return self._handle_blank(timestamp)
        
        # Filter low confidence
        if confidence < self.min_confidence:
            return self._handle_low_confidence(timestamp)
        
        # Update state
        self.state = DetectionState.DETECTING
        self.last_detection_time = timestamp
        
        # Same letter as current - accumulate
        if letter == self.current_letter:
            self.current_letter_frames += 1
            self.current_letter_confidence_sum += confidence
        else:
            # Different letter - finalize current and start new
            self._finalize_current_letter()
            self.current_letter = letter
            self.current_letter_frames = 1
            self.current_letter_confidence_sum = confidence
        
        return None
    
    def _handle_blank(self, timestamp: float) -> Optional[FingerspellingResult]:
        """Handle blank/pause detection"""
        self.blank_frames += 1
        
        # Finalize any pending letter
        self._finalize_current_letter()
        
        # Check if pause threshold exceeded
        if self.last_detection_time is not None:
            pause_duration = timestamp - self.last_detection_time
            
            if pause_duration > self.pause_threshold_ms and len(self.detected_letters) > 0:
                # Word complete
                self.state = DetectionState.COMPLETE
                return self.get_result()
        
        return None
    
    def _handle_low_confidence(self, timestamp: float) -> Optional[FingerspellingResult]:
        """Handle low confidence detection"""
        # Treat as potential pause
        return self._handle_blank(timestamp)
    
    def _finalize_current_letter(self):
        """Finalize the current letter if it meets criteria"""
        if self.current_letter is None:
            return
        
        if self.current_letter_frames >= self.min_hold_frames:
            avg_confidence = self.current_letter_confidence_sum / self.current_letter_frames
            
            if avg_confidence >= self.min_confidence:
                detection = LetterDetection(
                    letter=self.current_letter,
                    confidence=avg_confidence,
                    timestamp=time.time() * 1000,
                    frame_count=self.current_letter_frames
                )
                self.detected_letters.append(detection)
                logger.debug(f"Letter confirmed: {self.current_letter} (conf: {avg_confidence:.2f})")
        
        # Reset current letter tracking
        self.current_letter = None
        self.current_letter_frames = 0
        self.current_letter_confidence_sum = 0.0
    
    def force_complete(self) -> FingerspellingResult:
        """Force completion and return current result"""
        self._finalize_current_letter()
        self.state = DetectionState.COMPLETE
        return self.get_result()
    
    def get_result(self) -> FingerspellingResult:
        """Get the current detection result"""
        # Finalize any pending letter
        if self.current_letter is not None:
            self._finalize_current_letter()
        
        # Build result
        letters = [d.letter for d in self.detected_letters]
        confidences = [d.confidence for d in self.detected_letters]
        
        name = "".join(letters)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        detection_time = 0.0
        if self.start_time is not None and self.last_detection_time is not None:
            detection_time = self.last_detection_time - self.start_time
        
        return FingerspellingResult(
            detected_letters=letters,
            detected_name=name,
            confidence=avg_confidence,
            language=self.language,
            is_complete=(self.state == DetectionState.COMPLETE),
            detection_time_ms=detection_time,
            letter_confidences=confidences
        )
    
    def get_current_state(self) -> Dict:
        """Get current detection state for frontend updates"""
        result = self.get_result()
        
        return {
            "state": self.state.value,
            "current_letter": self.current_letter,
            "current_letter_frames": self.current_letter_frames,
            "detected_letters": result.detected_letters,
            "partial_name": result.detected_name,
            "total_frames": self.total_frames_processed,
            "blank_frames": self.blank_frames,
            "language": self.language
        }
    
    def set_language(self, language: str):
        """Change the active language and reset"""
        self.language = language.upper()
        self.reset()
        logger.info(f"Language changed to {self.language}")


class FingerspellingSession:
    """
    Manages a complete fingerspelling detection session.
    Combines alphabet classification with sequencing.
    """
    
    def __init__(self, language: str = "ASL"):
        from alphabet_classifier import get_classifier_manager
        
        self.language = language.upper()
        self.classifier_manager = get_classifier_manager()
        self.detector = FingerspellingDetector(language=self.language)
        
        self.is_active = False
        self.frame_count = 0
        
        logger.info(f"FingerspellingSession created for {self.language}")
    
    def start(self):
        """Start the detection session"""
        self.is_active = True
        self.detector.reset()
        self.frame_count = 0
        logger.info("Fingerspelling session started")
    
    def stop(self) -> FingerspellingResult:
        """Stop the session and return final result"""
        self.is_active = False
        result = self.detector.force_complete()
        logger.info(f"Fingerspelling session stopped. Result: {result.detected_name}")
        return result
    
    def process_landmarks(self, landmarks) -> Dict:
        """
        Process hand landmarks and return current state.
        
        Args:
            landmarks: Hand landmarks from MediaPipe (21 points with x, y, z)
        
        Returns:
            Current detection state dict
        """
        if not self.is_active:
            return {"error": "Session not active"}
        
        self.frame_count += 1
        
        # Classify the landmarks
        prediction = self.classifier_manager.classify(self.language, landmarks)
        
        # Add to detector
        result = self.detector.add_detection(
            letter=prediction.letter,
            confidence=prediction.confidence
        )
        
        # Get current state
        state = self.detector.get_current_state()
        state["last_prediction"] = {
            "letter": prediction.letter,
            "confidence": prediction.confidence,
            "is_blank": prediction.is_blank
        }
        state["frame_count"] = self.frame_count
        
        # Check if complete
        if result is not None:
            state["is_complete"] = True
            state["final_result"] = {
                "detected_letters": result.detected_letters,
                "detected_name": result.detected_name,
                "confidence": result.confidence
            }
        
        return state
    
    def set_language(self, language: str):
        """Change language and reset session"""
        self.language = language.upper()
        self.detector.set_language(language)
        self.frame_count = 0
    
    def get_language_info(self) -> Dict:
        """Get info about current language's alphabet"""
        return self.classifier_manager.get_language_info(self.language)


# Global session manager
_sessions: Dict[str, FingerspellingSession] = {}


def get_or_create_session(session_id: str, language: str = "ASL") -> FingerspellingSession:
    """Get existing session or create new one"""
    if session_id not in _sessions:
        _sessions[session_id] = FingerspellingSession(language)
    return _sessions[session_id]


def remove_session(session_id: str):
    """Remove a session"""
    if session_id in _sessions:
        del _sessions[session_id]
