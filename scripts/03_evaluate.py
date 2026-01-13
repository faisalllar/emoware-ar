"""
STEP 3: Evaluate Model
- Test accuracy
- Per-class metrics
- Confusion matrix
- Generate training curves
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import json
from pathlib import Path

DATA_PATH = Path("data")
MODELS_PATH = Path("models")
RESULTS_PATH = Path("results")

def load_data_and_model():
    """Load test data and trained model"""
    print("[*] Loading test data and model...")
    
    X_test = np.load(f"{DATA_PATH}/X_test.npy")
    y_test = np.load(f"{DATA_PATH}/y_test.npy")
    
    model = keras.models.load_model(f"{MODELS_PATH}/emotion_model_v2.h5")
    
    print(f"[+] X_test: {X_test.shape}")
    print(f"[+] y_test: {y_test.shape}")
    print(f"[+] Model loaded!")
    
    return X_test, y_test, model

def evaluate_model(model, X_test, y_test):
    """Evaluate on test set"""
    print("[*] Evaluating model...")
    
    # One-hot encode y_test
    y_test_oh = keras.utils.to_categorical(y_test, 4)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    
    print(f"[+] Test Loss: {test_loss:.4f}")
    print(f"[+] Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return test_loss, test_acc

def get_predictions(model, X_test):
    """Get predictions"""
    print("[*] Getting predictions...")
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    return y_pred

def generate_metrics(y_test, y_pred):
    """Generate per-class metrics"""
    print("[*] Generating metrics...")
    
    emotion_names = ['Anxious', 'Confused', 'Bored', 'Focused']
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=emotion_names, digits=4))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    
    print("\nPER-CLASS METRICS:")
    for i, emotion in enumerate(emotion_names):
        print(f"{emotion:10s} | P={precision[i]:.2%} | R={recall[i]:.2%} | F1={f1[i]:.2%}")
    
    return {
        'emotions': emotion_names,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }

def generate_confusion_matrix(y_test, y_pred):
    """Generate confusion matrix"""
    print("[*] Generating confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    emotion_names = ['Anxious', 'Confused', 'Bored', 'Focused']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/confusion_matrix.png", dpi=150)
    print(f"[+] Saved: {RESULTS_PATH}/confusion_matrix.png")
    
    return cm

def generate_training_curves():
    """Generate training curves from history"""
    print("[*] Generating training curves...")
    
    with open(f"{RESULTS_PATH}/training_history.json", 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/training_curves.png", dpi=150)
    print(f"[+] Saved: {RESULTS_PATH}/training_curves.png")

def save_metrics(metrics, test_loss, test_acc):
    """Save all metrics to JSON"""
    print("[*] Saving metrics...")
    
    metrics_dict = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_accuracy_percent': float(test_acc) * 100,
        **metrics
    }
    
    with open(f"{RESULTS_PATH}/performance_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"[+] Saved: {RESULTS_PATH}/performance_metrics.json")

def main():
    print("="*60)
    print("EMOWARE-AR: EVALUATE MODEL")
    print("="*60)
    
    # Load
    X_test, y_test, model = load_data_and_model()
    
    # Evaluate
    test_loss, test_acc = evaluate_model(model, X_test, y_test)
    
    # Get predictions
    y_pred = get_predictions(model, X_test)
    
    # Metrics
    metrics = generate_metrics(y_test, y_pred)
    
    # Confusion matrix
    cm = generate_confusion_matrix(y_test, y_pred)
    
    # Training curves
    generate_training_curves()
    
    # Save metrics
    save_metrics(metrics, test_loss, test_acc)
    
    print("\n[✓] Evaluation complete!")
    print("[✓] Graphs saved in results/")
    print("[✓] Next: Run 04_adaptive_loop.py")

if __name__ == "__main__":
    main()
