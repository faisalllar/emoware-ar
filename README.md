# EMOWARE-AR: Emotion-Aware Adaptive Learning System

## Overview
EMOWARE-AR is an intelligent adaptive learning system that uses real-time emotion detection via facial recognition to personalize the learning experience. The system monitors student emotions (Anxious, Confused, Bored, Focused) and dynamically adjusts content difficulty and presentation based on emotional feedback.

## Features
- **Emotion Detection**: CNN-based facial emotion recognition (4 classes)
- **Adaptive Learning Loop**: 5-step intelligent adaptation system
  1. Display Content (Level-based)
  2. Collect Feedback (Emotion + Quiz)
  3. Classify Feedback
  4. Update Material (Difficulty adjustment)
  5. Repeat/Progress
- **Performance Metrics**: Classification reports, confusion matrices, training curves
- **Session Logging**: Complete learning session tracking with emotion, performance, and adaptations

## Dataset
- **Source**: FER-2013 + Custom EMOWARE-AR labels
- **Classes**: Anxious, Confused, Bored, Focused
- **Train**: 28,709 images | **Test**: 7,178 images
- **Image Size**: 48×48 grayscale

## Model Performance
- **Test Accuracy**: 72.30%
- **Class Metrics**:
  - Anxious: P=70.87%, R=71.88%, F1=71.37%
  - Confused: P=80.41%, R=71.12%, F1=75.48%
  - Bored: P=85.19%, R=41.44%, F1=55.76%
  - Focused: P=71.68%, R=74.21%, F1=72.92%

## Project Structure
```
emoware-ar/
├── scripts/
│   ├── 01_data_prep.py          # Data preprocessing & loading
│   ├── 02_train_model.py        # CNN model training
│   ├── 03_evaluate.py           # Model evaluation & metrics
│   └── 04_adaptive_loop.py      # Adaptive learning system simulation
├── results/
│   ├── training_curves.png      # Loss & accuracy curves
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   ├── performance_metrics.json # Detailed metrics
│   └── learning_session_log.json # Session simulation log
├── convert_fer2013_to_csv.py    # FER-2013 data conversion utility
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Preprocess data (if needed)
python scripts/01_data_prep.py

# Train the model
python scripts/02_train_model.py

# Evaluate the model
python scripts/03_evaluate.py

# Run adaptive learning simulation
python scripts/04_adaptive_loop.py
```

### 3. View Results
All results are saved in the `results/` folder:
- Training curves: `results/training_curves.png`
- Confusion matrix: `results/confusion_matrix.png`
- Performance metrics: `results/performance_metrics.json`
- Session log: `results/learning_session_log.json`

## System Architecture

### CNN Model Architecture
```
Input (48×48×1)
  ↓
Conv2D (32 filters) → Conv2D (32 filters) → MaxPool → Dropout
  ↓
Conv2D (64 filters) → Conv2D (64 filters) → MaxPool → Dropout
  ↓
Conv2D (128 filters) → Conv2D (128 filters) → MaxPool → Dropout
  ↓
Flatten (4608 units)
  ↓
Dense (256) → BatchNorm → Dropout
  ↓
Dense (128) → BatchNorm → Dropout
  ↓
Dense (4) → Softmax
```
Total Parameters: 1,501,284

### Adaptive Learning Logic
The system implements a 5-step loop:

1. **Display Content**: Present material at current difficulty level
2. **Collect Feedback**: 
   - Detect student emotion from facial expression
   - Evaluate quiz answer (correct/incorrect)
3. **Classify Feedback**:
   - Map (emotion, performance) → action
   - Anxiety + Wrong → Anxiety Relief
   - Confusion + Wrong → Repeat Content
   - Correct Answer (3x) → Level Up
4. **Update Material**: Adjust difficulty, repeat content, or progress to next level
5. **Repeat**: Continue until mastery achieved or session ends

## Configuration
Content materials are organized by levels (1-5) with increasing complexity:
- **Level 1**: Pengenalan Kubus (Cube Introduction)
- **Level 2**: Sifat-Sifat Kubus (Cube Properties)
- **Level 3**: Volume & Luas Kubus (Volume & Surface Area)
- **Level 4**: Hubungan Antar Elemen (Relationships)
- **Level 5**: Masalah Kompleks (Complex Problems)

Each level includes:
- Title
- Explanation
- Quiz questions with correct answers
- Duration estimate

## Technologies
- **Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, Pillow
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Scikit-learn

## Future Improvements
- Real-time webcam integration for live emotion detection
- Multi-language support
- Subject diversification (beyond mathematics)
- Advanced reinforcement learning for adaptive strategy
- Integration with learning management systems (LMS)

## Author
Faisa - AI/ML Project

## License
Educational Use

---

**Project Status**: ✅ Complete
- ✅ Data preparation
- ✅ Model training (72.30% accuracy)
- ✅ Evaluation & metrics
- ✅ Adaptive learning system
- ✅ Documentation
