"""
STEP 2: Train CNN Model
- Build CNN architecture
- Train 25 epochs
- Save model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from pathlib import Path

DATA_PATH = Path("data")
MODELS_PATH = Path("models")
RESULTS_PATH = Path("results")

def load_data():
    """Load preprocessed data"""
    print("[*] Loading preprocessed data...")
    
    X_train = np.load(f"{DATA_PATH}/X_train.npy")
    y_train = np.load(f"{DATA_PATH}/y_train.npy")
    X_val = np.load(f"{DATA_PATH}/X_val.npy")
    y_val = np.load(f"{DATA_PATH}/y_val.npy")
    X_test = np.load(f"{DATA_PATH}/X_test.npy")
    y_test = np.load(f"{DATA_PATH}/y_test.npy")
    
    # One-hot encode
    NUM_CLASSES = 4
    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh = keras.utils.to_categorical(y_val, NUM_CLASSES)
    
    print(f"[+] X_train: {X_train.shape}")
    print(f"[+] y_train: {y_train_oh.shape}")
    print(f"[+] X_val: {X_val.shape}")
    print(f"[+] y_val: {y_val_oh.shape}")
    
    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test

def build_model(input_shape=(48, 48, 1), num_classes=4):
    """Build CNN architecture"""
    print("[*] Building CNN model...")
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train model"""
    print("[*] Starting training...")
    
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
    )
    
    return history

def save_model(model, model_path):
    """Save trained model"""
    print(f"[*] Saving model to {model_path}...")
    model.save(model_path)
    print("[+] Model saved!")

def save_history(history, output_path):
    """Save training history"""
    print(f"[*] Saving training history...")
    
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']]
    }
    
    # Add validation metrics if available
    if 'val_loss' in history.history:
        history_dict['val_loss'] = [float(x) for x in history.history['val_loss']]
    if 'val_accuracy' in history.history:
        history_dict['val_accuracy'] = [float(x) for x in history.history['val_accuracy']]
    
    with open(output_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print("[+] History saved!")

def main():
    print("="*60)
    print("EMOWARE-AR: TRAIN CNN MODEL")
    print("="*60)
    
    # Create output dirs
    MODELS_PATH.mkdir(exist_ok=True)
    RESULTS_PATH.mkdir(exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Build model
    model = build_model()
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Save
    save_model(model, f"{MODELS_PATH}/emotion_model_v2.h5")
    save_history(history, f"{RESULTS_PATH}/training_history.json")
    
    print("\n[✓] Training complete!")
    print("[✓] Model saved: models/emotion_model_v2.h5")
    print("[✓] History saved: results/training_history.json")
    print("[✓] Next: Run 03_evaluate.py")

if __name__ == "__main__":
    main()
