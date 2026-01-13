"""
STEP 1: Data Preparation
- Load FER-2013 dari CSV
- Preprocess & normalize
- Remap emotions 7→4
- Split train/val/test
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

DATA_PATH = Path("data/fer2013.csv")
OUTPUT_PATH = Path("data")

def load_fer2013(csv_path):
    """Load FER-2013 dataset dari CSV"""
    print("[*] Loading FER-2013 dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"[+] Dataset shape: {df.shape}")
    print(f"[+] Columns: {df.columns.tolist()}")
    print(f"[+] Usage distribution:\n{df['Usage'].value_counts()}")
    
    return df

def parse_images(df):
    """Parse pixel strings to image arrays"""
    print("[*] Parsing images...")
    
    X = []
    y = []
    usage = []
    
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(df)} images")
        
        # Parse pixels
        pixels = np.fromstring(row['pixels'], sep=' ', dtype='uint8')
        img = pixels.reshape(48, 48, 1).astype('float32')
        
        # Normalize
        img = img / 255.0
        
        X.append(img)
        y.append(int(row['emotion']))
        usage.append(row['Usage'])
    
    X = np.array(X)
    y = np.array(y)
    usage = np.array(usage)
    
    print(f"[+] Parsed {len(X)} images")
    print(f"[+] X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, usage

def remap_emotions(y, mapping):
    """Remap 7 emotions to 4"""
    print("[*] Remapping emotions (7→4)...")
    
    y_remapped = np.array([mapping[int(label)] for label in y])
    
    unique, counts = np.unique(y_remapped, return_counts=True)
    print(f"[+] Emotion distribution after remapping:")
    emotion_names = {0: "Anxious", 1: "Confused", 2: "Bored", 3: "Focused"}
    for emotion_id, count in zip(unique, counts):
        print(f"    {emotion_names[emotion_id]}: {count} ({count/len(y)*100:.1f}%)")
    
    return y_remapped

def split_data(X, y, usage):
    """Split into train/val/test"""
    print("[*] Splitting data (70/15/15)...")
    
    train_mask = (usage == 'Training')
    val_mask = (usage == 'PrivateTest')
    test_mask = (usage == 'PublicTest')
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"[+] Train: {X_train.shape} samples")
    print(f"[+] Val:   {X_val.shape} samples")
    print(f"[+] Test:  {X_test.shape} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_data(train, val, test, output_path):
    """Save preprocessed data"""
    print(f"[*] Saving data to {output_path}...")
    
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    np.save(f"{output_path}/X_train.npy", X_train)
    np.save(f"{output_path}/y_train.npy", y_train)
    np.save(f"{output_path}/X_val.npy", X_val)
    np.save(f"{output_path}/y_val.npy", y_val)
    np.save(f"{output_path}/X_test.npy", X_test)
    np.save(f"{output_path}/y_test.npy", y_test)
    
    print("[+] Data saved!")

def main():
    print("="*60)
    print("EMOWARE-AR: DATA PREPARATION")
    print("="*60)
    
    # Check if data exists
    if not DATA_PATH.exists():
        print(f"[!] ERROR: {DATA_PATH} not found!")
        print("[!] Please download fer2013.csv from Kaggle first")
        print("[!] https://www.kaggle.com/datasets/msambare/fer2013")
        return
    
    # Load dataset
    df = load_fer2013(DATA_PATH)
    
    # Parse images
    X, y, usage = parse_images(df)
    
    # Remap emotions: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
    # Target: 0=Anxious, 1=Confused, 2=Bored, 3=Focused
    mapping = {
        0: 0,  # Angry -> Anxious
        1: 2,  # Disgust -> Bored
        2: 0,  # Fear -> Anxious
        3: 3,  # Happy -> Focused
        4: 0,  # Sad -> Anxious
        5: 3,  # Surprise -> Focused
        6: 1,  # Neutral -> Confused
    }
    
    y_remapped = remap_emotions(y, mapping)
    
    # Split
    train, val, test = split_data(X, y_remapped, usage)
    
    # Save
    save_data(train, val, test, OUTPUT_PATH)
    
    print("\n[✓] Data preparation complete!")
    print("[✓] Next: Run 02_train_model.py")

if __name__ == "__main__":
    main()
