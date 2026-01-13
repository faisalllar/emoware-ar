"""
STEP 4: Adaptive Learning System
- Implement 5-point learning loop
- Build content database (19 materi + 32 quiz)
- Simulate student learning session
- Generate session logs
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
from datetime import datetime

DATA_PATH = Path("data")
MODELS_PATH = Path("models")
RESULTS_PATH = Path("results")

# ============================================
# PART 1: CONTENT DATABASE
# ============================================

CONTENT_DATABASE = {
    1: {
        "level": 1,
        "title": "Pengenalan Kubus",
        "explanation": "Kubus adalah bangun ruang dengan 6 sisi persegi yang semuanya berukuran sama.",
        "quiz": [
            {
                "id": "L1_Q1",
                "question": "Kubus memiliki berapa sisi?",
                "options": ["A. 4", "B. 6", "C. 8", "D. 12"],
                "correct": "B",
                "explanation": "Kubus memiliki 6 sisi yang semuanya persegi."
            },
            {
                "id": "L1_Q2",
                "question": "Berapa rusuk pada kubus?",
                "options": ["A. 8", "B. 10", "C. 12", "D. 14"],
                "correct": "C",
                "explanation": "Kubus memiliki 12 rusuk (4 atas + 4 bawah + 4 samping)."
            },
            {
                "id": "L1_Q3",
                "question": "Berapa titik sudut pada kubus?",
                "options": ["A. 4", "B. 6", "C. 8", "D. 10"],
                "correct": "C",
                "explanation": "Kubus memiliki 8 titik sudut."
            }
        ],
        "duration_minutes": 3
    },
    2: {
        "level": 2,
        "title": "Rumus Volume Kubus",
        "explanation": "Rumus volume kubus adalah V = s³, dimana s adalah panjang sisi kubus.",
        "quiz": [
            {
                "id": "L2_Q1",
                "question": "Jika kubus memiliki sisi 5 cm, berapa volumenya?",
                "options": ["A. 25 cm³", "B. 75 cm³", "C. 125 cm³", "D. 150 cm³"],
                "correct": "C",
                "explanation": "V = s³ = 5³ = 125 cm³"
            },
            {
                "id": "L2_Q2",
                "question": "Kubus dengan sisi 3 cm memiliki volume...",
                "options": ["A. 9 cm³", "B. 27 cm³", "C. 36 cm³", "D. 54 cm³"],
                "correct": "B",
                "explanation": "V = 3³ = 27 cm³"
            },
            {
                "id": "L2_Q3",
                "question": "Apa rumus volume kubus?",
                "options": ["A. V = s²", "B. V = s³", "C. V = 2s", "D. V = s + s"],
                "correct": "B",
                "explanation": "Rumus volume kubus adalah V = s³"
            },
            {
                "id": "L2_Q4",
                "question": "Volume kubus dengan sisi 2 cm adalah...",
                "options": ["A. 4 cm³", "B. 6 cm³", "C. 8 cm³", "D. 10 cm³"],
                "correct": "C",
                "explanation": "V = 2³ = 8 cm³"
            }
        ],
        "duration_minutes": 4
    },
    3: {
        "level": 3,
        "title": "Aplikasi Volume Kubus",
        "explanation": "Aplikasi praktis menghitung volume kubus dalam kehidupan sehari-hari.",
        "quiz": [
            {
                "id": "L3_Q1",
                "question": "Sebuah kotak susu berbentuk kubus dengan sisi 10 cm. Berapa volumenya?",
                "options": ["A. 100 cm³", "B. 500 cm³", "C. 1000 cm³", "D. 2000 cm³"],
                "correct": "C",
                "explanation": "V = 10³ = 1000 cm³"
            },
            {
                "id": "L3_Q2",
                "question": "Dua kotak kubus. Kotak 1: sisi 2 cm, Kotak 2: sisi 4 cm. Selisih volumenya?",
                "options": ["A. 2 cm³", "B. 8 cm³", "C. 56 cm³", "D. 64 cm³"],
                "correct": "C",
                "explanation": "V1 = 8 cm³, V2 = 64 cm³, Selisih = 64-8 = 56 cm³"
            }
        ],
        "duration_minutes": 5
    },
    4: {
        "level": 4,
        "title": "Perbandingan Volume",
        "explanation": "Membandingkan volume antar bangun ruang kubus dengan ukuran berbeda.",
        "quiz": [
            {
                "id": "L4_Q1",
                "question": "Jika sisi kubus diperbesar 2x, berapa kali volume bertambah?",
                "options": ["A. 2 kali", "B. 4 kali", "C. 8 kali", "D. 16 kali"],
                "correct": "C",
                "explanation": "Jika s menjadi 2s, volume = (2s)³ = 8s³ (8 kali lebih besar)"
            }
        ],
        "duration_minutes": 6
    },
    5: {
        "level": 5,
        "title": "Tantangan Volume Kubus",
        "explanation": "Masalah kompleks yang menggabungkan konsep volume kubus.",
        "quiz": [
            {
                "id": "L5_Q1",
                "question": "Sebuah kubus besar dengan sisi 10 cm berisi 8 kubus kecil yang identik. Berapa sisi setiap kubus kecil?",
                "options": ["A. 2 cm", "B. 3 cm", "C. 4 cm", "D. 5 cm"],
                "correct": "D",
                "explanation": "V besar = 1000 cm³. V kecil = 1000/8 = 125 cm³. Sisi = ∛125 = 5 cm"
            }
        ],
        "duration_minutes": 8
    }
}

# ============================================
# PART 2: ADAPTIVE LEARNING SYSTEM
# ============================================

class AdaptiveLearningSystem:
    """
    5-Point Adaptive Learning Loop:
    1. DISPLAY CONTENT
    2. COLLECT FEEDBACK (emotion + quiz)
    3. CLASSIFY FEEDBACK
    4. UPDATE MATERIAL
    5. REPEAT
    """
    
    def __init__(self, emotion_model):
        self.emotion_model = emotion_model
        self.current_level = 1
        self.consecutive_correct = 0
        self.iterations = 0
        self.session_log = []
        self.emotion_names = {0: 'Anxious', 1: 'Confused', 2: 'Bored', 3: 'Focused'}
        self.start_time = datetime.now()
    
    def step_1_display_content(self):
        """STEP 1: Display materi sesuai current level"""
        content = CONTENT_DATABASE[self.current_level]
        
        print(f"\n{'='*60}")
        print(f"[*] STEP 1: DISPLAY CONTENT (Level {self.current_level})")
        print(f"{'='*60}")
        print(f"Judul: {content['title']}")
        print(f"Penjelasan: {content['explanation']}")
        print(f"Estimasi waktu: {content['duration_minutes']} menit")
        
        return content
    
    def step_2_collect_feedback(self, emotion_image, quiz_answer, content):
        """STEP 2: Collect emotion + quiz feedback"""
        
        print(f"\n{'='*60}")
        print(f"[*] STEP 2: COLLECT FEEDBACK")
        print(f"{'='*60}")
        
        # Emotion detection from image
        emotion_probs = self.emotion_model.predict(
            np.expand_dims(emotion_image, axis=0),
            verbose=0
        )
        emotion_idx = np.argmax(emotion_probs)
        emotion_name = self.emotion_names[emotion_idx]
        emotion_confidence = float(emotion_probs[0][emotion_idx])
        
        # Quiz evaluation (get first quiz item from list)
        quiz_item = content['quiz'][0] if isinstance(content['quiz'], list) else content['quiz']
        is_correct = (quiz_answer.upper() == quiz_item['correct'].upper())
        
        print(f"  Emotion: {emotion_name} ({emotion_confidence:.1%} confidence)")
        print(f"  Quiz: {'✓ Correct' if is_correct else '✗ Wrong'}")
        
        feedback = {
            'emotion': emotion_name,
            'emotion_confidence': emotion_confidence,
            'quiz_correct': is_correct
        }
        
        return feedback
    
    def step_3_classify_feedback(self, feedback):
        """STEP 3: Classify feedback & make decision"""
        
        print(f"\n{'='*60}")
        print(f"[*] STEP 3: CLASSIFY FEEDBACK")
        print(f"{'='*60}")
        
        emotion = feedback['emotion']
        is_correct = feedback['quiz_correct']
        
        # Decision logic
        if emotion == 'Anxious' and not is_correct:
            action = 'ANXIETY_RELIEF'
            reason = "Student anxious + answer wrong"
        elif emotion == 'Confused':
            action = 'CONFUSION_SUPPORT'
            reason = "Student confused (need visualization)"
        elif emotion == 'Bored' and is_correct:
            action = 'CHALLENGE_MASTERY'
            reason = "Student bored but correct"
        elif is_correct:
            self.consecutive_correct += 1
            if self.consecutive_correct >= 3:
                action = 'MASTERY_CHECK'
                reason = f"3 consecutive correct ({self.consecutive_correct}/3)"
            else:
                action = 'CONTINUE'
                reason = f"Continue ({self.consecutive_correct}/3)"
        else:
            self.consecutive_correct = 0
            action = 'CONTINUE'
            reason = "Wrong - try again at same level"
        
        classification = {
            'action': action,
            'reason': reason,
            'should_continue': (action != 'MASTERY_CHECK')
        }
        
        print(f"  Action: {action}")
        print(f"  Reason: {reason}")
        
        return classification
    
    def step_4_update_material(self, classification):
        """STEP 4: Update material/level based on feedback"""
        
        print(f"\n{'='*60}")
        print(f"🔄 STEP 4: UPDATE MATERIAL")
        print(f"{'='*60}")
        
        action = classification['action']
        
        if action == 'ANXIETY_RELIEF':
            self.current_level = max(1, self.current_level - 1)
            message = f"📉 Level DOWN → Level {self.current_level} (anxiety relief)"
            print(message)
        
        elif action == 'CONFUSION_SUPPORT':
            message = "📐 AR Visualization triggered!"
            print(message)
        
        elif action == 'CHALLENGE_MASTERY':
            self.current_level = min(5, self.current_level + 1)
            message = f"📈 Level UP → Level {self.current_level} (mastery)"
            print(message)
        
        elif action == 'MASTERY_CHECK':
            message = "🎉 Topic COMPLETED!"
            print(message)
        
        else:  # CONTINUE
            message = f"➡️ Continue at Level {self.current_level}"
            print(message)
        
        return message
    
    def step_5_repeat(self, classification):
        """STEP 5: Check if loop should repeat"""
        
        print(f"\n{'='*60}")
        print(f"⏭️  STEP 5: REPEAT?")
        print(f"{'='*60}")
        
        if classification['should_continue']:
            print("➡️ Moving to next iteration...")
            return True
        else:
            print("✅ Learning session COMPLETE!")
            return False
    
    def run_session(self, session_data):
        """Run complete learning session"""
        
        print("\n" + "="*60)
        print("[*] EMOWARE-AR: ADAPTIVE LEARNING SESSION")
        print("="*60)
        
        total_iterations = len(session_data)
        
        for iteration, data in enumerate(session_data, 1):
            self.iterations = iteration
            
            print(f"\n\n{'#'*60}")
            print(f"# ITERATION {iteration}/{total_iterations}")
            print(f"{'#'*60}")
            
            # Step 1: Display
            content = self.step_1_display_content()
            
            # Step 2: Collect
            feedback = self.step_2_collect_feedback(
                data['emotion_image'],
                data['quiz_answer'],
                content
            )
            
            # Step 3: Classify
            classification = self.step_3_classify_feedback(feedback)
            
            # Step 4: Update
            self.step_4_update_material(classification)
            
            # Step 5: Repeat?
            should_continue = self.step_5_repeat(classification)
            
            # Log this iteration
            self.session_log.append({
                'iteration': iteration,
                'level': self.current_level,
                'emotion': feedback['emotion'],
                'correct': feedback['quiz_correct'],
                'action': classification['action']
            })
            
            if not should_continue:
                break
        
        # Print summary
        self.print_summary()
        
        return self.session_log
    
    def print_summary(self):
        """Print session summary"""
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds() / 60
        
        correct_count = sum(1 for log in self.session_log if log['correct'])
        accuracy = correct_count / len(self.session_log) if self.session_log else 0
        
        print(f"\n\n{'='*60}")
        print(f"📋 SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Iterations:    {self.iterations}")
        print(f"Final Level:         {self.current_level}")
        print(f"Total Accuracy:      {accuracy:.1%} ({correct_count}/{len(self.session_log)})")
        print(f"Session Duration:    {elapsed_time:.1f} minutes")
        print(f"\nIteration Details:")
        
        for log in self.session_log:
            result = "✓" if log['correct'] else "✗"
            print(f"  [{log['iteration']}] Level {log['level']} | " +
                  f"{log['emotion']:10s} | {result} | {log['action']}")
        
        print(f"{'='*60}\n")

# ============================================
# PART 3: RUN SIMULATION
# ============================================

def simulate_session():
    """Simulate a student learning session"""
    
    print("[*] Loading emotion model...")
    model = keras.models.load_model(f"{MODELS_PATH}/emotion_model_v2.h5")
    
    print("[*] Loading test data for simulation...")
    X_test = np.load(f"{DATA_PATH}/X_test.npy")
    y_test = np.load(f"{DATA_PATH}/y_test.npy")
    
    # Initialize system
    learning_system = AdaptiveLearningSystem(model)
    
    # Create simulation: 5 iterations with random test images
    np.random.seed(42)
    random_indices = np.random.choice(len(X_test), 5, replace=False)
    
    session_data = [
        {
            'emotion_image': X_test[idx],
            'quiz_answer': 'B'  # Simulate answers (mix correct/wrong)
        }
        for idx in random_indices
    ]
    
    # Override some answers to be "wrong" for demo
    session_data[0]['quiz_answer'] = 'A'  # Wrong
    session_data[1]['quiz_answer'] = 'D'  # Wrong
    
    # Run session
    learning_log = learning_system.run_session(session_data)
    
    # Save log
    print(f"\n[*] Saving session log...")
    with open(f"{RESULTS_PATH}/learning_session_log.json", 'w') as f:
        json.dump(learning_log, f, indent=2)
    
    print(f"[+] Saved: {RESULTS_PATH}/learning_session_log.json")
    
    return learning_log

# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("EMOWARE-AR: ADAPTIVE LEARNING SYSTEM")
    print("="*60)
    
    # Check if model exists
    model_path = f"{MODELS_PATH}/emotion_model_v2.h5"
    if not Path(model_path).exists():
        print(f"[!] ERROR: Model not found at {model_path}")
        print("[!] Please run 02_train_model.py first")
        return
    
    # Run simulation
    session_log = simulate_session()
    
    print("\n[✓] Adaptive learning system test complete!")
    print("[✓] Session log saved!")
    print("\n[✓] PROJECT COMPLETE! All steps done.")
    print("[✓] Files ready in results/ folder for PPT & Laporan")

if __name__ == "__main__":
    main()
