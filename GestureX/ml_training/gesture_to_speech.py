"""
Gesture to Speech Module

Captures hand gestures via camera and converts them to speech using TTS.
Uses MediaPipe's pre-trained gesture recognizer for robust recognition.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import time
import os


# Gesture to speech mapping
GESTURE_SPEECH_MAP = {
    "Closed_Fist": "Stop",
    "Open_Palm": "Hello",
    "Pointing_Up": "One moment please",
    "Thumb_Down": "No",
    "Thumb_Up": "Yes",
    "Victory": "Peace",
    "ILoveYou": "I love you",
}


class GestureToSpeech:
    """Converts hand gestures to speech."""
    
    def __init__(self, model_path="gesture_recognizer.task"):
        """Initialize the gesture recognizer and TTS engine."""
        self.model_path = model_path
        self.recognizer = None
        self.tts_engine = None
        self.last_gesture = None
        self.last_spoken_time = 0
        self.cooldown = 2.0  # seconds between speaking same gesture
        
        self._init_recognizer()
        self._init_tts()
    
    def _init_recognizer(self):
        """Initialize MediaPipe gesture recognizer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Download: curl -O https://storage.googleapis.com/mediapipe-models/"
                "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            )
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
    
    def _init_tts(self):
        """Initialize text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            self.tts_engine = None
    
    def recognize_gesture(self, frame):
        """
        Recognize gesture in frame.
        
        Returns:
            tuple: (gesture_name, confidence, landmarks)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.recognizer.recognize(mp_image)
        
        if result.gestures and len(result.gestures) > 0:
            gesture = result.gestures[0][0]
            landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
            return gesture.category_name, gesture.score, landmarks
        
        return "None", 0.0, None
    
    def speak(self, text):
        """Speak text with cooldown to avoid repetition."""
        if not self.tts_engine:
            return False
        
        current_time = time.time()
        
        if text == self.last_gesture:
            if current_time - self.last_spoken_time < self.cooldown:
                return False
        
        self.last_gesture = text
        self.last_spoken_time = current_time
        
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            print(f"TTS error: {e}")
            return False
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame."""
        if not landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw points
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)            # Palm
        ]
        
        for start, end in connections:
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame
    
    def draw_info(self, frame, gesture, confidence, speech_text):
        """Draw gesture info on frame."""
        # Background box
        cv2.rectangle(frame, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 140), (0, 255, 0), 2)
        
        # Text
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Speech: {speech_text}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'ESC' for menu", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Run gesture to speech recognition."""
        print("\n" + "="*50)
        print("   GESTURE → SPEECH MODE")
        print("="*50)
        print("\nShow these gestures to camera:")
        for gesture, speech in GESTURE_SPEECH_MAP.items():
            print(f"  • {gesture} → \"{speech}\"")
        print("\nPress 'q' to quit, 'ESC' to return to menu")
        print("="*50 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror
                
                # Recognize gesture
                gesture, confidence, landmarks = self.recognize_gesture(frame)
                
                # Get speech text
                speech_text = GESTURE_SPEECH_MAP.get(gesture, "")
                
                # Draw on frame
                if landmarks:
                    frame = self.draw_landmarks(frame, landmarks)
                frame = self.draw_info(frame, gesture, confidence, speech_text)
                
                # Speak if confident
                if confidence > 0.7 and speech_text:
                    self.speak(speech_text)
                
                cv2.imshow("Gesture to Speech", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return True
    
    def close(self):
        """Release resources."""
        if self.recognizer:
            self.recognizer.close()


def run_gesture_to_speech():
    """Entry point for gesture to speech mode."""
    try:
        g2s = GestureToSpeech()
        g2s.run()
        g2s.close()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return False
    return True


if __name__ == "__main__":
    run_gesture_to_speech()
