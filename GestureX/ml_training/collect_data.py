"""
collect_data.py - Gesture Data Collection Module

Captures hand landmarks using webcam + MediaPipe and saves to CSV.
Each sample contains 63 features (21 landmarks × 3 coordinates) + label.

Usage:
    python collect_data.py --gesture hello --samples 100 --output data/gestures.csv
    
Controls:
    's' - Start/Stop recording
    'q' - Quit
    'r' - Reset count
"""

import argparse
import os
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import save_landmark_row, get_dataset_info, ensure_dir, NUM_FEATURES


class HandLandmarkDetector:
    """MediaPipe-based hand landmark detector."""
    
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5
    ):
        """
        Initialize detector.
        
        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Detect hand and extract landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Tuple of (annotated_frame, landmarks_array or None, hand_detected)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        landmarks = None
        detected = False
        
        if results.multi_hand_landmarks:
            detected = True
            hand = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )
            
            # Extract flattened landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand.landmark
            ]).flatten().astype(np.float32)
        
        return frame, landmarks, detected
    
    def close(self):
        """Release resources."""
        self.hands.close()


class DataCollector:
    """
    High-level data collection interface for Streamlit app.
    
    Provides a simple API for collecting gesture samples programmatically.
    """
    
    def __init__(
        self,
        output_dir: str = 'data/raw',
        samples_per_gesture: int = 50
    ):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save CSV files
            samples_per_gesture: Default number of samples per gesture
        """
        self.output_dir = output_dir
        self.samples_per_gesture = samples_per_gesture
        ensure_dir(output_dir)
    
    def get_output_path(self, gesture_name: str) -> str:
        """Generate output CSV path for a gesture."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"gesture_{gesture_name}_{timestamp}.csv"
        return os.path.join(self.output_dir, filename)
    
    def collect(
        self,
        gesture_name: str,
        num_samples: Optional[int] = None,
        camera_id: int = 0
    ) -> int:
        """
        Collect gesture samples.
        
        Args:
            gesture_name: Name of the gesture
            num_samples: Number of samples (uses default if None)
            camera_id: Camera device ID
            
        Returns:
            Number of samples collected
        """
        if num_samples is None:
            num_samples = self.samples_per_gesture
        
        output_path = self.get_output_path(gesture_name)
        
        return collect_gesture_samples(
            gesture_name=gesture_name,
            num_samples=num_samples,
            output_path=output_path,
            camera_id=camera_id
        )


def collect_gesture_samples(
    gesture_name: str,
    num_samples: int = 100,
    output_path: str = 'data/gestures.csv',
    camera_id: int = 0,
    sample_delay: float = 0.1
) -> int:
    """
    Collect gesture samples from webcam.
    
    Args:
        gesture_name: Label for the gesture
        num_samples: Number of samples to collect
        output_path: Path to save CSV file
        camera_id: Camera device ID
        sample_delay: Delay between samples (seconds)
        
    Returns:
        Number of samples collected
    """
    # Ensure output directory exists
    ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else 'data')
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        print("Check if camera is connected and not used by another app.")
        return 0
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize detector
    detector = HandLandmarkDetector()
    
    collected = 0
    recording = False
    last_sample_time = 0
    
    print("\n" + "=" * 50)
    print(f"GESTURE DATA COLLECTION")
    print("=" * 50)
    print(f"Gesture: {gesture_name}")
    print(f"Target: {num_samples} samples")
    print(f"Output: {output_path}")
    print("\nControls:")
    print("  [S] Start/Stop recording")
    print("  [R] Reset count")
    print("  [Q] Quit")
    print("=" * 50 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame")
                continue
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Detect hand
            frame, landmarks, hand_detected = detector.detect(frame)
            
            # Draw UI
            status_color = (0, 255, 0) if recording else (0, 165, 255)
            status_text = "RECORDING" if recording else "PAUSED"
            
            # Header background
            cv2.rectangle(frame, (0, 0), (640, 90), (40, 40, 40), -1)
            
            # Status info
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {collected}/{num_samples}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Hand detection indicator
            hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            hand_text = "Hand OK" if hand_detected else "No Hand"
            cv2.putText(frame, hand_text, (520, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
            
            # Progress bar
            progress = collected / num_samples if num_samples > 0 else 0
            bar_w = 300
            cv2.rectangle(frame, (320, 60), (320 + bar_w, 80), (100, 100, 100), -1)
            cv2.rectangle(frame, (320, 60), (320 + int(bar_w * progress), 80), (0, 255, 0), -1)
            cv2.putText(frame, f"{int(progress*100)}%", (320 + bar_w + 10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Record if conditions met
            if recording and hand_detected and collected < num_samples:
                current_time = time.time()
                if current_time - last_sample_time >= sample_delay:
                    save_landmark_row(output_path, landmarks, gesture_name)
                    collected += 1
                    last_sample_time = current_time
                    
                    # Flash indicator
                    cv2.circle(frame, (600, 50), 15, (0, 255, 0), -1)
                    
                    if collected >= num_samples:
                        recording = False
                        print(f"\n✓ Collection complete! {collected} samples saved.")
            
            # Show frame
            cv2.imshow('Gesture Collection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                recording = not recording
                print(f"Recording: {'ON' if recording else 'OFF'}")
            elif key == ord('r'):
                collected = 0
                print("Count reset to 0")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
    
    return collected


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Collect gesture data')
    parser.add_argument('--gesture', '-g', type=str, required=True, help='Gesture name/label')
    parser.add_argument('--samples', '-n', type=int, default=100, help='Number of samples')
    parser.add_argument('--output', '-o', type=str, default='data/gestures.csv', help='Output CSV path')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera ID')
    parser.add_argument('--delay', '-d', type=float, default=0.1, help='Delay between samples')
    parser.add_argument('--info', action='store_true', help='Show dataset info and exit')
    
    args = parser.parse_args()
    
    if args.info:
        info = get_dataset_info(args.output)
        if info['exists']:
            print(f"\nDataset: {args.output}")
            print(f"Total samples: {info['total_samples']}")
            print(f"Classes: {info['num_classes']}")
            for cls, count in info['class_distribution'].items():
                print(f"  - {cls}: {count}")
        else:
            print(f"No dataset found at {args.output}")
        return
    
    count = collect_gesture_samples(
        gesture_name=args.gesture,
        num_samples=args.samples,
        output_path=args.output,
        camera_id=args.camera,
        sample_delay=args.delay
    )
    
    print(f"\nSummary: Collected {count} samples of '{args.gesture}'")
    
    # Show dataset info
    info = get_dataset_info(args.output)
    if info['exists']:
        print(f"Dataset now has {info['total_samples']} total samples")


if __name__ == '__main__':
    main()
