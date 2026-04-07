"""
model.py - Neural Network Model Definition

Defines MLP classifier for static gesture recognition.
Lightweight model designed to run on CPU.

Usage:
    from model import build_mlp_model, save_model, load_model
"""

import os
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utils import NUM_FEATURES


def build_mlp_model(
    num_classes: int,
    num_features: int = NUM_FEATURES,
    hidden_units: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    use_batchnorm: bool = True
) -> Model:
    """
    Build MLP model for gesture classification.
    
    Architecture: Input -> [Dense -> BN -> ReLU -> Dropout] x N -> Softmax
    
    Args:
        num_classes: Number of gesture classes
        num_features: Input feature dimension (default: 63)
        hidden_units: List of hidden layer sizes
        dropout_rate: Dropout rate
        use_batchnorm: Whether to use batch normalization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential(name='GestureClassifier')
    
    # Input layer
    model.add(layers.Input(shape=(num_features,), name='input'))
    
    # Hidden layers
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, name=f'dense_{i+1}'))
        if use_batchnorm:
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Activation('relu', name=f'relu_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_small_model(num_classes: int) -> Model:
    """Build smaller model for faster inference."""
    return build_mlp_model(
        num_classes=num_classes,
        hidden_units=[64, 32],
        dropout_rate=0.2,
        use_batchnorm=False
    )


def build_large_model(num_classes: int) -> Model:
    """Build larger model for better accuracy."""
    return build_mlp_model(
        num_classes=num_classes,
        hidden_units=[256, 128, 64, 32],
        dropout_rate=0.4,
        use_batchnorm=True
    )


def get_callbacks(
    model_path: str = 'models/gesture_model.h5',
    patience: int = 15
) -> List:
    """
    Get training callbacks.
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
    ]


def save_model(model: Model, path: str) -> None:
    """Save Keras model to file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


def load_model(path: str) -> Model:
    """Load Keras model from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = keras.models.load_model(path)
    print(f"Model loaded from {path}")
    return model


def get_model_summary(model: Model) -> str:
    """Get model summary as string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)


if __name__ == '__main__':
    print("Testing model.py...")
    
    # Test model building
    model = build_mlp_model(num_classes=10)
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print("\nModel test passed!")
