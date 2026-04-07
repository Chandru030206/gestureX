"""
inference.py - Live Gesture Recognition with Text-to-Speech

Real-time gesture detection using webcam, classifies gestures
and speaks detected labels using pyttsx3.

Usage:
    python inference.py --model models/gesture_model.h5
"""

import argparse
import os
import time
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
import mediapipe as mp
import pyttsx3

try:
    import tensorflow as tf
except ImportError:
    from tensorflow import keras as tf

from utils import load_label_encoder


# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GestureRecognizer:
    """Real-time gesture recognition with TTS."""
    
    def __init__(
        self,
        model_path: str = 'models/gesture_model.h5',
        encoder_path: str = 'models/label_encoder.pkl',
        confidence_threshold: float = 0.70,
        cooldown: float = 2.0,
        speak: bool = True
    ):
        """
        Initialize recognizer.
        
        Args:
            model_path: Path to trained model
            encoder_path: Path to label encoder
            confidence_threshold: Min confidence to accept prediction
            cooldown: Seconds between TTS announcements
            speak: Enable text-to-speech
        """
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded!")
        
        # Load encoder
        print(f"Loading encoder from {encoder_path}...")
        encoder_data = load_label_encoder(encoder_path)
        self.encoder = encoder_data['encoder']  # label -> int
        self.decoder = encoder_data['decoder']  # int -> label
        self.classes = list(self.encoder.keys())
        print(f"Classes: {self.classes}")
        
        # Settings
        self.threshold = confidence_threshold
        self.cooldown = cooldown
        self.speak_enabled = speak
        
        # TTS engine
        if speak:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
        else:
            self.engine = None
        
        # State
        self.last_spoken = ""
        self.last_spoken_time = 0.0
        self.last_gesture = ""
        self.last_confidence = 0.0
        
        # Mediapipe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Array of 63 normalized features or None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        hand = results.multi_hand_landmarks[0]
        
        # Get wrist position for normalization
        wrist = hand.landmark[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z
        
        # Extract and normalize landmarks
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x - wx, lm.y - wy, lm.z - wz])
        
        return np.array(landmarks, dtype=np.float32)
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture from landmarks.
        
        Args:
            landmarks: Array of 63 features
            
        Returns:
            (gesture_label, confidence)
        """
        # Reshape for model
        features = landmarks.reshape(1, -1)
        
        # Predict
        probs = self.model.predict(features, verbose=0)[0]
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        
        label = self.decoder.get(class_idx, "UNKNOWN")
        
        return label, confidence
    
    def speak(self, text: str) -> None:
        """Speak text using TTS."""
        if not self.speak_enabled or not self.engine:
            return
        
        current_time = time.time()
        
        # Check cooldown and avoid repeating
        if (text != self.last_spoken or 
            current_time - self.last_spoken_time > self.cooldown):
            
            print(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            
            self.last_spoken = text
            self.last_spoken_time = current_time
    
    def draw_overlay(
        self,
        frame: np.ndarray,
        gesture: str = "",
        confidence: float = 0.0,
        hand_results: Any = None
    ) -> np.ndarray:
        """
        Draw visualization overlay on frame.
        
        Args:
            frame: BGR image
            gesture: Detected gesture label
            confidence: Prediction confidence
            hand_results: Mediapipe hand results
            
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        
        # Draw hand landmarks
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Status box
        box_h = 100
        cv2.rectangle(frame, (0, 0), (w, box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, box_h), (255, 255, 255), 2)
        
        # Gesture text
        if gesture and confidence >= self.threshold:
            color = (0, 255, 0)  # Green for confident
            text = f"{gesture} ({confidence*100:.1f}%)"
        elif gesture:
            color = (0, 165, 255)  # Orange for low confidence
            text = f"{gesture}? ({confidence*100:.1f}%)"
        else:
            color = (128, 128, 128)
            text = "No hand detected"
        
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, color, 3)
        
        # Threshold indicator
        cv2.putText(frame, f"Threshold: {self.threshold*100:.0f}%", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions
        instructions = "Press: Q=Quit | S=Toggle Speech | +/-=Adjust Threshold"
        cv2.putText(frame, instructions, (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self, camera_index: int = 0) -> None:
        """
        Run live gesture recognition.
        
        Args:
            camera_index: Camera device index
        """
        print(f"\nStarting camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "=" * 50)
        print("LIVE GESTURE RECOGNITION")
        print("=" * 50)
        print(f"Classes: {self.classes}")
        print(f"Threshold: {self.threshold*100:.0f}%")
        print(f"Speech: {'ON' if self.speak_enabled else 'OFF'}")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Toggle speech")
        print("  +/- - Adjust threshold")
        print("=" * 50 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed!")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                gesture = ""
                confidence = 0.0
                
                # Extract landmarks and predict
                landmarks = self.extract_landmarks(frame)
                if landmarks is not None:
                    gesture, confidence = self.predict(landmarks)
                    
                    self.last_gesture = gesture
                    self.last_confidence = confidence
                    
                    # Speak if confident
                    if confidence >= self.threshold:
                        self.speak(gesture)
                
                # Draw overlay
                frame = self.draw_overlay(frame, gesture, confidence, results)
                
                # Display
                cv2.imshow('Gesture Recognition', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("Exiting...")
                    break
                elif key == ord('s'):
                    self.speak_enabled = not self.speak_enabled
                    status = "ON" if self.speak_enabled else "OFF"
                    print(f"Speech: {status}")
                elif key == ord('+') or key == ord('='):
                    self.threshold = min(0.99, self.threshold + 0.05)
                    print(f"Threshold: {self.threshold*100:.0f}%")
                elif key == ord('-') or key == ord('_'):
                    self.threshold = max(0.10, self.threshold - 0.05)
                    print(f"Threshold: {self.threshold*100:.0f}%")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("Cleanup complete.")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Live gesture recognition')
    parser.add_argument('--model', '-m', type=str, default='models/gesture_model.h5',
                       help='Model path')
    parser.add_argument('--encoder', '-e', type=str, default='models/label_encoder.pkl',
                       help='Encoder path')
    parser.add_argument('--threshold', '-t', type=float, default=0.70,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--no-speak', action='store_true',
                       help='Disable text-to-speech')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("Train a model first: python train.py")
        return
    
    if not os.path.exists(args.encoder):
        print(f"ERROR: Encoder not found: {args.encoder}")
        return
    
    recognizer = GestureRecognizer(
        model_path=args.model,
        encoder_path=args.encoder,
        confidence_threshold=args.threshold,
        speak=not args.no_speak
    )
    
    recognizer.run(camera_index=args.camera)


if __name__ == '__main__':
    main()
