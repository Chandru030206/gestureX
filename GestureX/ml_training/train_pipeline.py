"""
Training Pipeline for GestureX Duo
====================================
Collects MediaPipe landmark data, trains MLP classifiers, exports to ONNX.

Usage:
    # Collect data for ASL word gestures
    python train_pipeline.py collect --language ASL --type word --gesture HELLO --samples 200

    # Train word model for ASL
    python train_pipeline.py train --language ASL --type word

    # Train alphabet model for ASL
    python train_pipeline.py train --language ASL --type alphabet

    # Export trained model to ONNX
    python train_pipeline.py export --language ASL --type word

Rules:
    - Never share weights or datasets between languages
    - Each language has separate word_model and alphabet_model
    - Minimum 200 samples per class
    - Single hand: 63 features, Two hands: 126 features
    - All landmarks normalized relative to wrist (index 0)
"""

import os
import sys
import csv
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Language Configuration ───
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

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def get_data_path(language: str, model_type: str) -> Path:
    """Get CSV data path for a language/type combination."""
    path = DATA_DIR / "landmarks" / language / f"{model_type}_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_model_path(language: str, model_type: str) -> Path:
    """Get model output path."""
    path = MODELS_DIR / language / f"{model_type}_model.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ═══════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════

def collect_data(language: str, model_type: str, gesture_label: str, num_samples: int = 200):
    """
    Collect landmark data using webcam and MediaPipe.
    
    Saves to CSV: each row = [feature_0, feature_1, ..., feature_N, label]
    Single hand: 63 columns + label
    Two hands: 126 columns + label
    """
    import cv2
    import mediapipe as mp

    config = LANGUAGE_CONFIG[language]
    num_hands = config["hands"]
    num_features = config["features"]

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    data_path = get_data_path(language, model_type)
    
    # Check if file exists to determine if we need header
    file_exists = data_path.exists()
    
    collected = 0
    logger.info(f"Collecting {num_samples} samples for '{gesture_label}' ({language} {model_type})")
    logger.info(f"Features: {num_features} ({num_hands} hand{'s' if num_hands > 1 else ''})")
    logger.info("Press 'c' to capture, 'q' to quit")
    logger.info("─" * 50)

    with open(data_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            header = [f"f{i}" for i in range(num_features)] + ["label"]
            writer.writerow(header)

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Display info
            cv2.putText(frame, f"Language: {language} | Type: {model_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture_label}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {collected}/{num_samples}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, "Press 'c' to capture | 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("GestureX Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                if results.multi_hand_landmarks:
                    features = extract_landmarks(results.multi_hand_landmarks, num_hands, num_features)
                    if features is not None:
                        row = features.tolist() + [gesture_label]
                        writer.writerow(row)
                        collected += 1
                        logger.info(f"  ✓ Sample {collected}/{num_samples} captured")
                else:
                    logger.warning("  ✗ No hand detected — try again")

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"✓ Collected {collected} samples for '{gesture_label}' → {data_path}")


def collect_data_auto(language: str, model_type: str, gesture_label: str, 
                      num_samples: int = 200, interval_ms: int = 100):
    """
    Auto-collect landmark data at regular intervals (no manual trigger needed).
    Hold the gesture steady and samples are captured automatically.
    """
    import cv2
    import mediapipe as mp

    config = LANGUAGE_CONFIG[language]
    num_hands = config["hands"]
    num_features = config["features"]

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    data_path = get_data_path(language, model_type)
    file_exists = data_path.exists()

    collected = 0
    recording = False
    last_capture = 0

    logger.info(f"Auto-collect mode: {num_samples} samples for '{gesture_label}'")
    logger.info("Press 's' to start/stop recording, 'q' to quit")

    with open(data_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"f{i}" for i in range(num_features)] + ["label"]
            writer.writerow(header)

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            status = "● RECORDING" if recording else "○ PAUSED"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"{status} | {gesture_label} | {collected}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Auto-capture during recording
            if recording and results.multi_hand_landmarks:
                now_ms = int(time.time() * 1000)
                if now_ms - last_capture >= interval_ms:
                    features = extract_landmarks(results.multi_hand_landmarks, num_hands, num_features)
                    if features is not None:
                        row = features.tolist() + [gesture_label]
                        writer.writerow(row)
                        collected += 1
                        last_capture = now_ms

            cv2.imshow("GestureX Auto-Collect", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                recording = not recording
                logger.info(f"Recording {'started' if recording else 'paused'}")

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"✓ Collected {collected} samples → {data_path}")


def extract_landmarks(multi_hand_landmarks, num_hands: int, num_features: int) -> Optional[np.ndarray]:
    """Extract and normalize landmarks from MediaPipe results."""
    all_features = []

    for i, hand_lms in enumerate(multi_hand_landmarks):
        if i >= num_hands:
            break
        
        coords = []
        for lm in hand_lms.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        pts = np.array(coords, dtype=np.float32).reshape(-1, 3)
        
        # Normalize relative to wrist
        wrist = pts[0].copy()
        pts = pts - wrist
        max_val = np.max(np.abs(pts))
        if max_val > 0:
            pts = pts / max_val
        
        all_features.extend(pts.flatten().tolist())

    # Pad if needed (e.g., 2-hand language but only 1 hand detected)
    while len(all_features) < num_features:
        all_features.append(0.0)

    return np.array(all_features[:num_features], dtype=np.float32)


# ═══════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════

def train_model(language: str, model_type: str, epochs: int = 100, batch_size: int = 32):
    """
    Train a lightweight MLP classifier for the given language/type.
    
    Architecture: Input → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dense(num_classes, Softmax)
    Max 3 layers as specified.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    config = LANGUAGE_CONFIG[language]
    num_features = config["features"]
    
    data_path = get_data_path(language, model_type)
    if not data_path.exists():
        logger.error(f"No training data found: {data_path}")
        logger.info(f"Run: python train_pipeline.py collect --language {language} --type {model_type} --gesture <NAME>")
        return None
    
    # Load CSV data
    logger.info(f"Loading data from {data_path}…")
    import pandas as pd
    df = pd.read_csv(data_path)
    
    if len(df) < 10:
        logger.error(f"Not enough data ({len(df)} rows). Need at least 200 per class.")
        return None
    
    # Split features and labels
    X = df.iloc[:, :num_features].values.astype(np.float32)
    y = df.iloc[:, -1].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    labels = list(le.classes_)
    
    logger.info(f"Classes: {labels}")
    logger.info(f"Samples: {len(X)} | Features: {num_features} | Classes: {num_classes}")
    
    # Check minimum samples per class
    for cls in labels:
        count = np.sum(y == cls)
        if count < 200:
            logger.warning(f"  ⚠ Class '{cls}' has only {count} samples (recommended: 200+)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Build MLP model (max 3 layers)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Train
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"✓ Test accuracy: {accuracy:.4f}")
    
    # Save Keras model temporarily
    keras_path = MODELS_DIR / language / f"{model_type}_model.keras"
    model.save(str(keras_path))
    logger.info(f"✓ Keras model saved: {keras_path}")
    
    # Save labels
    labels_path = MODELS_DIR / language / f"{model_type}_labels.txt"
    with open(labels_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    logger.info(f"✓ Labels saved: {labels_path}")
    
    # Export to ONNX
    export_to_onnx(language, model_type)
    
    return {
        "language": language,
        "type": model_type,
        "accuracy": accuracy,
        "num_classes": num_classes,
        "labels": labels,
        "samples": len(X),
    }


# ═══════════════════════════════════════
# ONNX EXPORT
# ═══════════════════════════════════════

def export_to_onnx(language: str, model_type: str):
    """Convert a trained Keras model to ONNX format."""
    try:
        import tf2onnx
        import tensorflow as tf

        config = LANGUAGE_CONFIG[language]
        num_features = config["features"]
        
        keras_path = MODELS_DIR / language / f"{model_type}_model.keras"
        onnx_path = get_model_path(language, model_type)
        
        if not keras_path.exists():
            logger.error(f"Keras model not found: {keras_path}")
            return
        
        model = tf.keras.models.load_model(str(keras_path))
        
        # Convert to ONNX
        import onnx
        spec = (tf.TensorSpec((None, num_features), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
        
        onnx.save(model_proto, str(onnx_path))
        logger.info(f"✓ ONNX model exported: {onnx_path}")
        
    except ImportError as e:
        logger.warning(f"tf2onnx not installed. Install with: pip install tf2onnx")
        logger.warning("Skipping ONNX export. The Keras model is still saved.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GestureX Duo Training Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Collect data
    collect_parser = subparsers.add_parser("collect", help="Collect landmark data")
    collect_parser.add_argument("--language", required=True, choices=list(LANGUAGE_CONFIG.keys()))
    collect_parser.add_argument("--type", required=True, choices=["word", "alphabet"])
    collect_parser.add_argument("--gesture", required=True, help="Gesture/letter label")
    collect_parser.add_argument("--samples", type=int, default=200)
    collect_parser.add_argument("--auto", action="store_true", help="Auto-capture mode")

    # Train model
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--language", required=True, choices=list(LANGUAGE_CONFIG.keys()))
    train_parser.add_argument("--type", required=True, choices=["word", "alphabet"])
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)

    # Export to ONNX
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--language", required=True, choices=list(LANGUAGE_CONFIG.keys()))
    export_parser.add_argument("--type", required=True, choices=["word", "alphabet"])

    args = parser.parse_args()

    if args.command == "collect":
        if args.auto:
            collect_data_auto(args.language, args.type, args.gesture, args.samples)
        else:
            collect_data(args.language, args.type, args.gesture, args.samples)
    elif args.command == "train":
        result = train_model(args.language, args.type, args.epochs, args.batch_size)
        if result:
            logger.info(f"Training complete: {result}")
    elif args.command == "export":
        export_to_onnx(args.language, args.type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
