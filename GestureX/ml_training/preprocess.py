"""
preprocess.py - Data Preprocessing Module

Loads CSV dataset, normalizes landmarks, augments data, and splits into train/test sets.

Usage:
    python preprocess.py --input data/gestures.csv --output data/processed
"""

import argparse
import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    load_dataset_csv, 
    create_label_encoder, 
    encode_labels,
    save_label_encoder,
    ensure_dir,
    get_dataset_info,
    NUM_FEATURES
)


def normalize_to_wrist(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks relative to wrist (landmark 0).
    Makes gestures position-invariant.
    
    Args:
        landmarks: Array of shape (N, 63) or (63,)
        
    Returns:
        Normalized landmarks
    """
    landmarks = np.atleast_2d(landmarks).copy()
    
    for i in range(len(landmarks)):
        wrist = landmarks[i, :3].copy()  # x, y, z of wrist
        for j in range(21):
            landmarks[i, j*3:j*3+3] -= wrist
    
    return np.squeeze(landmarks)


def normalize_scale(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks by hand size (palm width).
    Makes gestures scale-invariant.
    
    Args:
        landmarks: Array of shape (N, 63) or (63,)
        
    Returns:
        Scale-normalized landmarks
    """
    landmarks = np.atleast_2d(landmarks).copy()
    
    for i in range(len(landmarks)):
        pts = landmarks[i].reshape(21, 3)
        
        # Use distance from wrist to middle finger MCP as scale reference
        wrist = pts[0]
        mcp = pts[9]  # Middle finger MCP
        scale = np.linalg.norm(mcp - wrist)
        
        if scale > 1e-6:
            landmarks[i] = (pts / scale).flatten()
    
    return np.squeeze(landmarks)


def augment_with_noise(landmarks: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to landmarks."""
    noise = np.random.normal(0, noise_std, landmarks.shape)
    return landmarks + noise


def augment_with_scale(landmarks: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Apply random scaling to landmarks."""
    landmarks = np.atleast_2d(landmarks).copy()
    for i in range(len(landmarks)):
        scale = np.random.uniform(*scale_range)
        landmarks[i] *= scale
    return np.squeeze(landmarks)


def augment_with_rotation(landmarks: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Apply random 2D rotation (x-y plane) to landmarks."""
    landmarks = np.atleast_2d(landmarks).copy()
    
    for i in range(len(landmarks)):
        angle = np.radians(np.random.uniform(-max_angle, max_angle))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        for j in range(21):
            x, y = landmarks[i, j*3], landmarks[i, j*3+1]
            landmarks[i, j*3] = x * cos_a - y * sin_a
            landmarks[i, j*3+1] = x * sin_a + y * cos_a
    
    return np.squeeze(landmarks)


def augment_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    factor: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment dataset with noise, scaling, and rotation.
    
    Args:
        X: Features array
        y: Labels array
        factor: Number of augmented copies per sample
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    all_X = [X]
    all_y = [y]
    
    for _ in range(factor):
        aug_X = X.copy()
        
        # Apply augmentations randomly
        aug_X = augment_with_noise(aug_X, noise_std=0.015)
        
        if np.random.random() > 0.5:
            aug_X = augment_with_scale(aug_X, (0.9, 1.1))
        
        if np.random.random() > 0.5:
            aug_X = augment_with_rotation(aug_X, max_angle=8.0)
        
        all_X.append(aug_X)
        all_y.append(y)
    
    return np.vstack(all_X), np.concatenate(all_y)


def preprocess_data(
    input_path: str,
    output_dir: str = 'data/processed',
    test_size: float = 0.2,
    augment: bool = True,
    augment_factor: int = 2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline.
    
    Args:
        input_path: Path to input CSV
        output_dir: Directory to save processed data
        test_size: Fraction for test set
        augment: Whether to augment data
        augment_factor: Augmentation multiplier
        random_state: Random seed
        
    Returns:
        Dictionary with preprocessing results
    """
    print("=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)
    
    # Load data
    print(f"\nLoading data from {input_path}...")
    X, labels = load_dataset_csv(input_path)
    print(f"Loaded {len(X)} samples")
    
    # Create label encoder
    encoder, decoder = create_label_encoder(labels)
    y = encode_labels(labels, encoder)
    num_classes = len(encoder)
    
    print(f"Classes ({num_classes}): {list(encoder.keys())}")
    
    # Normalize landmarks
    print("\nNormalizing landmarks...")
    X = normalize_to_wrist(X)
    X = normalize_scale(X)
    
    # Augment data
    if augment:
        print(f"Augmenting data (factor: {augment_factor})...")
        orig_size = len(X)
        X, y = augment_dataset(X, y, factor=augment_factor)
        print(f"Dataset size: {orig_size} -> {len(X)}")
    
    # Split data
    print(f"\nSplitting data (test_size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save processed data
    ensure_dir(output_dir)
    
    print(f"\nSaving to {output_dir}...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save scaler parameters
    np.save(os.path.join(output_dir, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(output_dir, 'scaler_scale.npy'), scaler.scale_)
    
    # Save metadata
    metadata = {
        'num_features': NUM_FEATURES,
        'num_classes': num_classes,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'classes': list(encoder.keys()),
        'encoder': encoder,
        'decoder': decoder
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {NUM_FEATURES}")
    print(f"Classes: {num_classes}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': metadata,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }


def load_processed_data(data_dir: str = 'data/processed') -> Dict[str, Any]:
    """Load preprocessed data from directory."""
    return {
        'X_train': np.load(os.path.join(data_dir, 'X_train.npy')),
        'X_test': np.load(os.path.join(data_dir, 'X_test.npy')),
        'y_train': np.load(os.path.join(data_dir, 'y_train.npy')),
        'y_test': np.load(os.path.join(data_dir, 'y_test.npy')),
        'metadata': np.load(os.path.join(data_dir, 'metadata.npy'), allow_pickle=True).item(),
        'scaler_mean': np.load(os.path.join(data_dir, 'scaler_mean.npy')),
        'scaler_scale': np.load(os.path.join(data_dir, 'scaler_scale.npy'))
    }


def preprocess_dataset(
    input_dir: str = 'data/raw',
    output_dir: str = 'data/processed',
    test_size: float = 0.2,
    augment: bool = True,
    augment_factor: int = 2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocess all CSV files in a directory.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save processed data
        test_size: Fraction for test set
        augment: Whether to augment data
        augment_factor: Augmentation multiplier
        random_state: Random seed
        
    Returns:
        Dictionary with preprocessing results
    """
    import pandas as pd
    
    print("=" * 50)
    print("DATASET PREPROCESSING")
    print("=" * 50)
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    print(f"\nFound {len(csv_files)} CSV files in {input_dir}")
    
    # Load and concatenate all data
    all_landmarks = []
    all_labels = []
    
    for filename in csv_files:
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(filepath)
            
            # Get landmark columns
            lm_cols = [col for col in df.columns if col.startswith('lm_')]
            
            if lm_cols and 'label' in df.columns:
                landmarks = df[lm_cols].values.astype(np.float32)
                labels = df['label'].tolist()
                
                all_landmarks.append(landmarks)
                all_labels.extend(labels)
                
                print(f"  Loaded {len(df)} samples from {filename}")
        except Exception as e:
            print(f"  Warning: Could not load {filename}: {e}")
    
    if not all_landmarks:
        raise ValueError("No valid data found in CSV files")
    
    X = np.vstack(all_landmarks)
    
    print(f"\nTotal: {len(X)} samples")
    
    # Create label encoder
    encoder, decoder = create_label_encoder(all_labels)
    y = encode_labels(all_labels, encoder)
    num_classes = len(encoder)
    
    print(f"Classes ({num_classes}): {list(encoder.keys())}")
    
    # Normalize landmarks
    print("\nNormalizing landmarks...")
    X = normalize_to_wrist(X)
    X = normalize_scale(X)
    
    # Augment data
    if augment:
        print(f"Augmenting data (factor: {augment_factor})...")
        orig_size = len(X)
        X, y = augment_dataset(X, y, factor=augment_factor)
        print(f"Dataset size: {orig_size} -> {len(X)}")
    
    # Split data
    print(f"\nSplitting data (test_size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save processed data
    ensure_dir(output_dir)
    
    print(f"\nSaving to {output_dir}...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save scaler parameters
    np.save(os.path.join(output_dir, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(output_dir, 'scaler_scale.npy'), scaler.scale_)
    
    # Save metadata
    metadata = {
        'num_features': NUM_FEATURES,
        'num_classes': num_classes,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(X_train) + len(X_test),
        'classes': list(encoder.keys()),
        'encoder': encoder,
        'decoder': decoder
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {NUM_FEATURES}")
    print(f"Classes: {num_classes}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': metadata,
        'total_samples': len(X_train) + len(X_test),
        'num_classes': num_classes
    }


def preprocess_single(
    landmarks: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray
) -> np.ndarray:
    """
    Preprocess a single sample for inference.
    
    Args:
        landmarks: Raw 63-feature landmarks
        scaler_mean: Scaler mean from training
        scaler_scale: Scaler scale from training
        
    Returns:
        Preprocessed features ready for model
    """
    # Normalize
    landmarks = normalize_to_wrist(landmarks)
    landmarks = normalize_scale(landmarks)
    
    # Standardize
    landmarks = (landmarks - scaler_mean) / scaler_scale
    
    return landmarks.reshape(1, -1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Preprocess gesture data')
    parser.add_argument('--input', '-i', type=str, default='data/gestures.csv', help='Input CSV')
    parser.add_argument('--output', '-o', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--augment-factor', type=int, default=2, help='Augmentation factor')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return
    
    preprocess_data(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        augment=not args.no_augment,
        augment_factor=args.augment_factor
    )


if __name__ == '__main__':
    main()
