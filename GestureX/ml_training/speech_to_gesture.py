"""
Speech to Gesture Module

Converts spoken or typed text to gesture visualization.
User can speak (using speech recognition) or type text,
and the system shows the corresponding gesture.
"""

import cv2
import numpy as np
import os
import time

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Note: speech_recognition not installed. Using text input only.")


# Speech/text to gesture mapping (reverse of gesture_to_speech)
SPEECH_GESTURE_MAP = {
    # Common phrases mapped to gestures
    "stop": "Closed_Fist",
    "wait": "Closed_Fist",
    "halt": "Closed_Fist",
    
    "hello": "Open_Palm",
    "hi": "Open_Palm",
    "hey": "Open_Palm",
    "bye": "Open_Palm",
    "goodbye": "Open_Palm",
    
    "wait": "Pointing_Up",
    "one moment": "Pointing_Up",
    "hold on": "Pointing_Up",
    "one": "Pointing_Up",
    
    "no": "Thumb_Down",
    "bad": "Thumb_Down",
    "dislike": "Thumb_Down",
    "wrong": "Thumb_Down",
    
    "yes": "Thumb_Up",
    "ok": "Thumb_Up",
    "okay": "Thumb_Up",
    "good": "Thumb_Up",
    "great": "Thumb_Up",
    "like": "Thumb_Up",
    "correct": "Thumb_Up",
    
    "peace": "Victory",
    "victory": "Victory",
    "two": "Victory",
    
    "i love you": "ILoveYou",
    "love": "ILoveYou",
    "love you": "ILoveYou",
}

# Gesture descriptions for display
GESTURE_INFO = {
    "Closed_Fist": {
        "name": "Closed Fist",
        "emoji": "✊",
        "description": "Make a fist with all fingers closed",
        "meaning": "Stop / Wait"
    },
    "Open_Palm": {
        "name": "Open Palm",
        "emoji": "✋",
        "description": "Open hand with all fingers extended",
        "meaning": "Hello / Goodbye"
    },
    "Pointing_Up": {
        "name": "Pointing Up",
        "emoji": "☝️",
        "description": "Index finger pointing up, others closed",
        "meaning": "One moment / Attention"
    },
    "Thumb_Down": {
        "name": "Thumb Down",
        "emoji": "👎",
        "description": "Thumb pointing down, fingers closed",
        "meaning": "No / Disagree"
    },
    "Thumb_Up": {
        "name": "Thumb Up", 
        "emoji": "👍",
        "description": "Thumb pointing up, fingers closed",
        "meaning": "Yes / OK / Good"
    },
    "Victory": {
        "name": "Victory / Peace",
        "emoji": "✌️",
        "description": "Index and middle fingers up in V shape",
        "meaning": "Peace / Victory / Two"
    },
    "ILoveYou": {
        "name": "I Love You",
        "emoji": "🤟",
        "description": "Thumb, index, and pinky extended",
        "meaning": "I Love You (ASL)"
    }
}


def draw_gesture_visualization(gesture_name, width=640, height=480):
    """
    Create a visualization of the gesture.
    
    Args:
        gesture_name: Name of the gesture to visualize
        width: Image width
        height: Image height
        
    Returns:
        numpy array: BGR image with gesture visualization
    """
    # Create blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background
    
    info = GESTURE_INFO.get(gesture_name, {})
    
    if not info:
        # Unknown gesture
        cv2.putText(img, "Unknown gesture", (width//2 - 150, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 2)
        return img
    
    # Draw title
    cv2.putText(img, "SPEECH -> GESTURE", (width//2 - 140, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw box
    box_x, box_y = 50, 70
    box_w, box_h = width - 100, height - 140
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
    
    # Draw emoji (large)
    emoji_text = info.get('emoji', '?')
    # Note: OpenCV doesn't render emojis well, so we'll use text representation
    cv2.putText(img, f"[ {info.get('name', gesture_name)} ]", (width//2 - 120, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Draw gesture name
    cv2.putText(img, gesture_name, (width//2 - 80, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Draw description
    desc = info.get('description', '')
    cv2.putText(img, "How to make:", (80, 260),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(img, desc, (80, 295),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw meaning
    meaning = info.get('meaning', '')
    cv2.putText(img, f"Meaning: {meaning}", (80, 340),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw hand diagram based on gesture
    draw_hand_diagram(img, gesture_name, width//2, 420)
    
    return img


def draw_hand_diagram(img, gesture_name, center_x, center_y):
    """Draw a simple hand diagram for the gesture."""
    
    # Palm center
    palm = (center_x, center_y - 30)
    
    # Finger positions (simplified)
    # [thumb, index, middle, ring, pinky]
    finger_tips = [
        (center_x - 60, center_y - 80),   # Thumb
        (center_x - 30, center_y - 100),  # Index
        (center_x, center_y - 110),       # Middle
        (center_x + 30, center_y - 100),  # Ring
        (center_x + 55, center_y - 85),   # Pinky
    ]
    
    finger_bases = [
        (center_x - 30, center_y - 20),   # Thumb base
        (center_x - 20, center_y - 30),   # Index base
        (center_x, center_y - 35),        # Middle base
        (center_x + 20, center_y - 30),   # Ring base
        (center_x + 35, center_y - 25),   # Pinky base
    ]
    
    # Define which fingers are up for each gesture
    fingers_up = {
        "Closed_Fist": [False, False, False, False, False],
        "Open_Palm": [True, True, True, True, True],
        "Pointing_Up": [False, True, False, False, False],
        "Thumb_Down": [True, False, False, False, False],  # Special case
        "Thumb_Up": [True, False, False, False, False],
        "Victory": [False, True, True, False, False],
        "ILoveYou": [True, True, False, False, True],
    }
    
    config = fingers_up.get(gesture_name, [False] * 5)
    
    # Draw palm
    cv2.ellipse(img, palm, (40, 50), 0, 0, 360, (150, 150, 200), -1)
    cv2.ellipse(img, palm, (40, 50), 0, 0, 360, (0, 255, 0), 2)
    
    # Draw fingers
    for i, (tip, base, is_up) in enumerate(zip(finger_tips, finger_bases, config)):
        if is_up:
            # Extended finger
            cv2.line(img, base, tip, (150, 150, 200), 12)
            cv2.line(img, base, tip, (0, 255, 0), 2)
            cv2.circle(img, tip, 8, (0, 255, 0), -1)
        else:
            # Closed finger (short)
            mid = ((base[0] + tip[0])//2, (base[1] + tip[1])//2 + 15)
            cv2.line(img, base, mid, (150, 150, 200), 10)
            cv2.line(img, base, mid, (0, 255, 0), 2)
    
    # Special handling for thumb down
    if gesture_name == "Thumb_Down":
        thumb_down_tip = (center_x - 60, center_y + 40)
        cv2.line(img, finger_bases[0], thumb_down_tip, (150, 150, 200), 12)
        cv2.line(img, finger_bases[0], thumb_down_tip, (0, 255, 0), 2)
        cv2.circle(img, thumb_down_tip, 8, (0, 255, 0), -1)


class SpeechToGesture:
    """Converts speech or text to gesture visualization."""
    
    def __init__(self):
        """Initialize speech recognition if available."""
        self.recognizer = None
        self.microphone = None
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            try:
                self.microphone = sr.Microphone()
                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception as e:
                print(f"Microphone not available: {e}")
                self.microphone = None
    
    def find_gesture(self, text):
        """
        Find matching gesture for text input.
        
        Args:
            text: Input text/speech
            
        Returns:
            str: Gesture name or None
        """
        text = text.lower().strip()
        
        # Direct match
        if text in SPEECH_GESTURE_MAP:
            return SPEECH_GESTURE_MAP[text]
        
        # Partial match
        for phrase, gesture in SPEECH_GESTURE_MAP.items():
            if phrase in text or text in phrase:
                return gesture
        
        return None
    
    def listen_for_speech(self, timeout=5):
        """
        Listen for speech input.
        
        Args:
            timeout: Maximum time to listen
            
        Returns:
            str: Recognized text or None
        """
        if not self.recognizer or not self.microphone:
            return None
        
        try:
            with self.microphone as source:
                print("Listening... (speak now)")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            text = self.recognizer.recognize_google(audio)
            print(f"Heard: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("No speech detected")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
        
        return None
    
    def run(self):
        """Run speech to gesture mode."""
        print("\n" + "="*50)
        print("   SPEECH → GESTURE MODE")
        print("="*50)
        print("\nSay or type these words to see gestures:")
        print("-" * 40)
        
        # Group by gesture
        gesture_words = {}
        for word, gesture in SPEECH_GESTURE_MAP.items():
            if gesture not in gesture_words:
                gesture_words[gesture] = []
            gesture_words[gesture].append(word)
        
        for gesture, words in gesture_words.items():
            info = GESTURE_INFO.get(gesture, {})
            emoji = info.get('emoji', '')
            print(f"  {emoji} {gesture}: {', '.join(words[:3])}")
        
        print("-" * 40)
        print("\nControls:")
        print("  • Type text and press Enter")
        if self.microphone:
            print("  • Press 's' to speak")
        print("  • Press 'q' or ESC to quit")
        print("="*50 + "\n")
        
        # Create initial window
        current_gesture = None
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        
        cv2.putText(img, "SPEECH -> GESTURE", (180, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, "Type in terminal or press 's' to speak", (100, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(img, "Press 'q' to quit", (230, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow("Speech to Gesture", img)
        
        # Input loop
        import threading
        import queue
        
        input_queue = queue.Queue()
        running = True
        
        def input_thread():
            while running:
                try:
                    text = input("\nEnter text (or 's' to speak, 'q' to quit): ")
                    input_queue.put(text)
                    if text.lower() == 'q':
                        break
                except EOFError:
                    break
        
        # Start input thread
        thread = threading.Thread(target=input_thread, daemon=True)
        thread.start()
        
        try:
            while running:
                # Check for keyboard input in OpenCV window
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('q') or key == 27:  # q or ESC
                    running = False
                    break
                
                if key == ord('s') and self.microphone:
                    # Speech input
                    cv2.putText(img, "Listening...", (250, 430),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("Speech to Gesture", img)
                    cv2.waitKey(1)
                    
                    text = self.listen_for_speech()
                    if text:
                        gesture = self.find_gesture(text)
                        if gesture:
                            current_gesture = gesture
                            img = draw_gesture_visualization(gesture)
                            # Add heard text
                            cv2.putText(img, f"Heard: \"{text}\"", (80, 380),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
                        else:
                            img = draw_gesture_visualization(None)
                            cv2.putText(img, f"No gesture for: \"{text}\"", (150, 250),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                    
                    cv2.imshow("Speech to Gesture", img)
                
                # Check for text input from terminal
                try:
                    text = input_queue.get_nowait()
                    
                    if text.lower() == 'q':
                        running = False
                        break
                    
                    if text.lower() == 's' and self.microphone:
                        # Speech mode
                        speech_text = self.listen_for_speech()
                        if speech_text:
                            text = speech_text
                        else:
                            continue
                    
                    gesture = self.find_gesture(text)
                    if gesture:
                        current_gesture = gesture
                        img = draw_gesture_visualization(gesture)
                        cv2.putText(img, f"Input: \"{text}\"", (80, 380),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
                    else:
                        img = draw_gesture_visualization(None)
                        cv2.putText(img, f"No gesture for: \"{text}\"", (150, 250),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                    
                    cv2.imshow("Speech to Gesture", img)
                    
                except queue.Empty:
                    pass
        
        finally:
            running = False
            cv2.destroyAllWindows()
        
        return True


def run_speech_to_gesture():
    """Entry point for speech to gesture mode."""
    s2g = SpeechToGesture()
    return s2g.run()


if __name__ == "__main__":
    run_speech_to_gesture()
