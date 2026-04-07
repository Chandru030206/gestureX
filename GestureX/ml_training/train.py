"""
train.py - Model Training Script

Trains gesture classifier, saves model + label encoder, displays metrics.

Usage:
    python train.py --data data/processed --output models/gesture_model.h5
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from model import build_mlp_model, save_model, get_callbacks
from preprocess import load_processed_data
from utils import save_label_encoder, ensure_dir


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    model_path: str = 'models/gesture_model.h5',
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.15
) -> Dict[str, Any]:
    """
    Train the gesture classifier.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_classes: Number of classes
        model_path: Path to save model
        epochs: Max epochs
        batch_size: Batch size
        validation_split: Validation fraction
        
    Returns:
        Dictionary with training results
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    print(f"\nDataset:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {num_classes}")
    
    # Build model
    print("\nBuilding model...")
    model = build_mlp_model(num_classes=num_classes)
    print(f"Parameters: {model.count_params():,}")
    
    # Callbacks
    ensure_dir(os.path.dirname(model_path) if os.path.dirname(model_path) else 'models')
    callbacks = get_callbacks(model_path=model_path)
    
    # Train
    print(f"\nTraining for up to {epochs} epochs...")
    print("-" * 60)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    print("-" * 60)
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    return {
        'model': model,
        'history': history.history,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'y_pred': y_pred,
        'y_true': y_test,
        'epochs_trained': len(history.history['loss'])
    }


def plot_history(history: Dict, output_path: str) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['loss'], 'b-', label='Train')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['accuracy'], 'b-', label='Train')
    if 'val_accuracy' in history:
        axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else '.')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training plot saved to {output_path}")


def plot_confusion(cm: np.ndarray, labels: list, output_path: str) -> None:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True',
        xlabel='Predicted',
        title='Confusion Matrix'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Text annotations
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else '.')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def run_training(
    data_dir: str = 'data/processed',
    model_path: str = 'models/gesture_model.h5',
    encoder_path: str = 'models/label_encoder.pkl',
    epochs: int = 100,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Run complete training pipeline.
    
    Args:
        data_dir: Preprocessed data directory
        model_path: Output model path
        encoder_path: Output encoder path
        epochs: Max training epochs
        batch_size: Batch size
        
    Returns:
        Training results
    """
    start = datetime.now()
    
    # Load data
    print("Loading preprocessed data...")
    data = load_processed_data(data_dir)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    metadata = data['metadata']
    
    num_classes = metadata['num_classes']
    classes = metadata['classes']
    encoder = metadata['encoder']
    decoder = metadata['decoder']
    
    print(f"Classes: {classes}")
    
    # Train
    results = train_model(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save encoder
    save_label_encoder(encoder, decoder, encoder_path)
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(results['y_true'], results['y_pred'], 
                                target_names=classes, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save plots
    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else 'models'
    plot_history(results['history'], os.path.join(model_dir, 'training_history.png'))
    plot_confusion(cm, classes, os.path.join(model_dir, 'confusion_matrix.png'))
    
    # Summary
    elapsed = datetime.now() - start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed}")
    print(f"Epochs: {results['epochs_trained']}")
    print(f"Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"Model: {model_path}")
    print(f"Encoder: {encoder_path}")
    print("=" * 60)
    
    results['metadata'] = metadata
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Train gesture model')
    parser.add_argument('--data', '-d', type=str, default='data/processed', help='Processed data dir')
    parser.add_argument('--output', '-o', type=str, default='models/gesture_model.h5', help='Model path')
    parser.add_argument('--encoder', '-e', type=str, default='models/label_encoder.pkl', help='Encoder path')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"ERROR: Data directory not found: {args.data}")
        print("Run preprocess.py first.")
        return
    
    run_training(
        data_dir=args.data,
        model_path=args.output,
        encoder_path=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
