"""
utils.py - Utility Functions for Gesture-Speech Engine

This module provides helper functions for:
- Label encoding/decoding
- CSV/JSON file operations
- Gesture vocabulary management
- Data validation utilities
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib


# =============================================================================
# Constants
# =============================================================================

NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63

# Column names for CSV
LANDMARK_COLUMNS = [f'lm_{i}_{c}' for i in range(NUM_LANDMARKS) for c in ['x', 'y', 'z']]


# =============================================================================
# Label Encoding/Decoding
# =============================================================================

def create_label_encoder(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional label encoder mappings.
    
    Args:
        labels: List of label strings
        
    Returns:
        Tuple of (label_to_int, int_to_label) dictionaries
    """
    unique_labels = sorted(set(labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    return label_to_int, int_to_label


def encode_labels(labels: List[str], encoder: Dict[str, int]) -> np.ndarray:
    """Encode string labels to integers."""
    return np.array([encoder[label] for label in labels], dtype=np.int32)


def decode_label(encoded: int, decoder: Dict[int, str]) -> str:
    """Decode single integer label to string."""
    return decoder[int(encoded)]


def save_label_encoder(encoder: Dict[str, int], decoder: Dict[int, str], path: str) -> None:
    """Save label encoder to file using joblib."""
    data = {'encoder': encoder, 'decoder': decoder}
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    joblib.dump(data, path)
    print(f"Label encoder saved to {path}")


def load_label_encoder(path: str) -> Dict[str, Any]:
    """
    Load label encoder from file.
    
    Returns:
        Dictionary with 'encoder' and 'decoder' keys
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoder file not found: {path}")
    data = joblib.load(path)
    return {'encoder': data['encoder'], 'decoder': data['decoder']}


# =============================================================================
# CSV Operations
# =============================================================================

def create_csv_columns() -> List[str]:
    """Get column names for gesture CSV file."""
    return LANDMARK_COLUMNS + ['label', 'timestamp']


def save_landmark_row(filepath: str, landmarks: np.ndarray, label: str) -> None:
    """
    Save a single landmark sample to CSV.
    
    Args:
        filepath: Path to CSV file
        landmarks: Flattened array of 63 landmark values
        label: Gesture label string
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    row_data = list(landmarks) + [label, timestamp]
    
    columns = create_csv_columns()
    df = pd.DataFrame([row_data], columns=columns)
    
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


def load_dataset_csv(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load landmarks and labels from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (landmarks array [N, 63], labels list)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    landmarks = df[LANDMARK_COLUMNS].values.astype(np.float32)
    labels = df['label'].tolist()
    
    return landmarks, labels


def get_dataset_info(filepath: str) -> Dict[str, Any]:
    """Get information about dataset file."""
    if not os.path.exists(filepath):
        return {'exists': False, 'path': filepath}
    
    df = pd.read_csv(filepath)
    label_counts = df['label'].value_counts().to_dict()
    
    return {
        'exists': True,
        'path': filepath,
        'total_samples': len(df),
        'num_classes': len(label_counts),
        'class_distribution': label_counts,
        'classes': list(label_counts.keys())
    }


# =============================================================================
# JSON Operations (Gestures Vocabulary)
# =============================================================================

def save_gestures_json(gestures: Dict, filepath: str) -> None:
    """
    Save gestures vocabulary to JSON file.
    
    Args:
        gestures: Dictionary with gesture data
                  Format: {"GESTURE_NAME": {"description": "...", "examples": [[63 floats], ...]}}
        filepath: Path to JSON file
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(gestures, f, indent=2, ensure_ascii=False)
    print(f"Gestures saved to {filepath}")


def load_gestures_json(filepath: str) -> Dict:
    """Load gestures vocabulary from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Gestures file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_default_gestures() -> Dict:
    """Get default gestures vocabulary structure."""
    return {
        "HELLO": {
            "speech": "Hello! Nice to meet you.",
            "examples": []
        },
        "GOODBYE": {
            "speech": "Goodbye! See you later.",
            "examples": []
        },
        "THANKS": {
            "speech": "Thank you very much!",
            "examples": []
        },
        "YES": {
            "speech": "Yes, I agree.",
            "examples": []
        },
        "NO": {
            "speech": "No, I disagree.",
            "examples": []
        },
        "HELP": {
            "speech": "I need help please.",
            "examples": []
        },
        "OK": {
            "speech": "Okay, understood.",
            "examples": []
        },
        "PEACE": {
            "speech": "Peace!",
            "examples": []
        },
        "THUMBS_UP": {
            "speech": "Great job!",
            "examples": []
        },
        "STOP": {
            "speech": "Please stop.",
            "examples": []
        }
    }


def export_csv_to_gestures_json(csv_path: str, json_path: str, max_examples: int = 5) -> None:
    """
    Export gesture examples from CSV to gestures.json format.
    
    Args:
        csv_path: Path to collected CSV data
        json_path: Path to output JSON file
        max_examples: Maximum examples per gesture to store
    """
    landmarks, labels = load_dataset_csv(csv_path)
    
    # Try to load existing gestures or create new
    try:
        gestures = load_gestures_json(json_path)
    except FileNotFoundError:
        gestures = {}
    
    # Group by label
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        examples = landmarks[indices[:max_examples]].tolist()
        
        if label.upper() not in gestures:
            gestures[label.upper()] = {
                "speech": label.replace('_', ' ').title(),
                "examples": examples
            }
        else:
            gestures[label.upper()]["examples"] = examples
    
    save_gestures_json(json_path, gestures)
    print(f"Exported {len(unique_labels)} gestures to {json_path}")


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_landmarks(landmarks: np.ndarray) -> bool:
    """Check if landmarks array has correct shape."""
    if landmarks is None:
        return False
    flat = landmarks.flatten()
    return flat.shape[0] == NUM_FEATURES


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence * 100:.1f}%"


def get_confidence_status(confidence: float, threshold: float) -> Tuple[str, str]:
    """
    Get confidence status and color.
    
    Returns:
        Tuple of (status_text, color_code)
    """
    if confidence >= threshold:
        return "HIGH", "#28a745"
    elif confidence >= threshold * 0.7:
        return "MEDIUM", "#ffc107"
    else:
        return "LOW", "#dc3545"


def suggest_improvement(confidence: float, threshold: float) -> Optional[str]:
    """Suggest improvements when confidence is low."""
    if confidence >= threshold:
        return None
    
    suggestions = [
        "Hold your hand more steadily",
        "Ensure good lighting",
        "Make sure your full hand is visible",
        "Perform the gesture more clearly",
        "Try moving closer to the camera"
    ]
    
    if confidence < 0.3:
        return suggestions[2]
    elif confidence < 0.5:
        return suggestions[3]
    else:
        return suggestions[0]


# =============================================================================
# Directory Utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path)


def get_project_paths() -> Dict[str, str]:
    """Get standard project paths."""
    return {
        'data_dir': 'data',
        'models_dir': 'models',
        'dataset_csv': 'data/gestures.csv',
        'processed_dir': 'data/processed',
        'model_path': 'models/gesture_model.h5',
        'encoder_path': 'models/label_encoder.pkl',
        'gestures_json': 'gestures.json'
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing utils.py...")
    
    # Test label encoding
    labels = ['hello', 'goodbye', 'hello', 'thanks']
    enc, dec = create_label_encoder(labels)
    print(f"Encoder: {enc}")
    print(f"Decoder: {dec}")
    
    # Test default gestures
    gestures = get_default_gestures()
    print(f"Default gestures: {list(gestures.keys())}")
    
    # Test paths
    paths = get_project_paths()
    print(f"Project paths: {paths}")
    
    print("\nAll tests passed!")
