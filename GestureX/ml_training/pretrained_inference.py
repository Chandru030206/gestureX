"""
Pre-trained Gesture Recognition using MediaPipe Gesture Recognizer.

This module uses Google's pre-trained gesture recognizer model which can detect:
- Closed_Fist
- Open_Palm
- Pointing_Up
- Thumb_Down
- Thumb_Up
- Victory (Peace sign)
- ILoveYou
- None (no gesture detected)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import time
import os


# Gesture to speech mapping
GESTURE_TO_SPEECH = {
    "Closed_Fist": "Stop",
    "Open_Palm": "Hello",
    "Pointing_Up": "One moment please",
    "Thumb_Down": "No",
    "Thumb_Up": "Yes, OK",
    "Victory": "Peace",
    "ILoveYou": "I love you",
    "None": ""
}


class PretrainedGestureRecognizer:
    """Wrapper for MediaPipe's pre-trained gesture recognizer."""
    
    def __init__(self, model_path="gesture_recognizer.task"):
        """
        Initialize the gesture recognizer.
        
        Args:
            model_path: Path to the gesture_recognizer.task file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            )
        
        # Create recognizer options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.last_gesture = None
        self.last_gesture_time = 0
        self.cooldown = 2.0  # seconds between speaking same gesture
        
        # Initialize TTS
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_available = True
        except Exception as e:
            print(f"TTS not available: {e}")
            self.tts_available = False
    
    def recognize(self, frame):
        """
        Recognize gesture in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            tuple: (gesture_name, confidence, handedness, landmarks)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Recognize
        result = self.recognizer.recognize(mp_image)
        
        if result.gestures and len(result.gestures) > 0:
            gesture = result.gestures[0][0]
            gesture_name = gesture.category_name
            confidence = gesture.score
            
            # Get handedness
            handedness = "Unknown"
            if result.handedness and len(result.handedness) > 0:
                handedness = result.handedness[0][0].category_name
            
            # Get landmarks
            landmarks = None
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                landmarks = result.hand_landmarks[0]
            
            return gesture_name, confidence, handedness, landmarks
        
        return "None", 0.0, "Unknown", None
    
    def speak(self, gesture_name):
        """Speak the gesture if available and not on cooldown."""
        if not self.tts_available:
            return
        
        current_time = time.time()
        
        # Check cooldown
        if gesture_name == self.last_gesture:
            if current_time - self.last_gesture_time < self.cooldown:
                return
        
        # Get speech text
        speech_text = GESTURE_TO_SPEECH.get(gesture_name, "")
        
        if speech_text:
            self.last_gesture = gesture_name
            self.last_gesture_time = current_time
            try:
                self.tts_engine.say(speech_text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame."""
        if landmarks is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                x1 = int(landmarks[start].x * w)
                y1 = int(landmarks[start].y * h)
                x2 = int(landmarks[end].x * w)
                y2 = int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame
    
    def close(self):
        """Release resources."""
        self.recognizer.close()


def run_pretrained_demo():
    """Run the pre-trained gesture recognition demo."""
    print("=" * 50)
    print("Pre-trained Gesture Recognition Demo")
    print("=" * 50)
    print("\nSupported Gestures:")
    for gesture, speech in GESTURE_TO_SPEECH.items():
        if speech:
            print(f"  {gesture}: \"{speech}\"")
    print("\nPress 'q' to quit")
    print("=" * 50)
    
    # Initialize
    recognizer = PretrainedGestureRecognizer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Recognize gesture
            gesture_name, confidence, handedness, landmarks = recognizer.recognize(frame)
            
            # Draw landmarks
            if landmarks:
                frame = recognizer.draw_landmarks(frame, landmarks)
            
            # Draw info
            cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 130), (0, 255, 0), 2)
            
            cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Hand: {handedness}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Speech text
            speech = GESTURE_TO_SPEECH.get(gesture_name, "")
            if speech:
                cv2.putText(frame, f"Speech: {speech}", (20, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Speak if confident
            if confidence > 0.7 and gesture_name != "None":
                recognizer.speak(gesture_name)
            
            cv2.imshow("Pre-trained Gesture Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()


if __name__ == "__main__":
    run_pretrained_demo()
