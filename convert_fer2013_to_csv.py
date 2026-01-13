"""
Convert FER-2013 folder structure (test/train with emotion subfolders) 
to single CSV file format
"""

import os
import csv
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
TRAIN_PATH = Path("data/train")
TEST_PATH = Path("data/test")
OUTPUT_CSV = Path("data/fer2013.csv")

# Emotion mapping (folder name → emotion ID)
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

def image_to_pixels_string(image_path):
    """Convert image to pixel string (48x48 grayscale)"""
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.resize((48, 48))
        pixels = np.array(img).flatten()
        return ' '.join(map(str, pixels))
    except Exception as e:
        print(f"  [!] Error processing {image_path}: {e}")
        return None

def scan_directory(base_path, usage_label):
    """Scan directory and return list of (pixels_string, emotion_id, usage)"""
    rows = []
    
    if not base_path.exists():
        print(f"[!] {base_path} not found!")
        return rows
    
    print(f"[*] Scanning {usage_label}...")
    
    total_count = 0
    for emotion_folder in base_path.iterdir():
        if not emotion_folder.is_dir():
            continue
        
        emotion_name = emotion_folder.name.lower()
        
        if emotion_name not in EMOTION_MAP:
            print(f"  [!] Unknown emotion: {emotion_name}, skipping...")
            continue
        
        emotion_id = EMOTION_MAP[emotion_name]
        image_count = 0
        
        for image_file in emotion_folder.glob('*.jpg'):
            pixels_string = image_to_pixels_string(image_file)
            
            if pixels_string:
                rows.append({
                    'emotion': emotion_id,
                    'pixels': pixels_string,
                    'Usage': usage_label
                })
                image_count += 1
                total_count += 1
        
        print(f"  ✓ {emotion_name}: {image_count} images")
    
    print(f"[+] Total {usage_label}: {total_count} images")
    return rows

def main():
    print("="*60)
    print("CONVERT FER-2013 FOLDER → CSV")
    print("="*60)
    
    # Create output directory
    Path("data").mkdir(exist_ok=True)
    
    # Scan train and test
    all_rows = []
    all_rows.extend(scan_directory(TRAIN_PATH, 'Training'))
    all_rows.extend(scan_directory(TEST_PATH, 'PublicTest'))
    
    if not all_rows:
        print("[!] No images found! Check folder structure.")
        return
    
    # Write CSV
    print(f"\n[*] Writing to {OUTPUT_CSV}...")
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['emotion', 'pixels', 'Usage'])
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"[+] CSV created: {OUTPUT_CSV}")
    print(f"[+] Total rows: {len(all_rows)}")
    print(f"\n[✓] Done!")
    print(f"[✓] Next: python scripts/01_data_prep.py")

if __name__ == "__main__":
    main()
