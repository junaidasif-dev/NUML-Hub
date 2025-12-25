"""
Main Runner Script for Speech Recognition System
Provides a menu-driven interface for all functionality
"""

import os
import sys


def print_header():
    """Print application header."""
    print("\n" + "="*60)
    print("     REAL-TIME SPEECH RECOGNITION SYSTEM")
    print("     Recognizing Spoken Digits (Zero to Nine)")
    print("="*60)


def print_menu():
    """Print main menu."""
    print("\nMain Menu:")
    print("-" * 40)
    print("  [1] Test Microphone")
    print("  [2] Record Dataset")
    print("  [3] Quick Record (3 samples/digit)")
    print("  [4] Train Model")
    print("  [5] Start Real-Time Recognition")
    print("  [6] Test Single Recognition")
    print("  [7] View System Info")
    print("  [q] Quit")
    print("-" * 40)


def test_microphone():
    """Test microphone functionality."""
    from record_dataset import check_microphone
    check_microphone()


def record_full_dataset():
    """Record the full dataset."""
    from record_dataset import record_dataset
    record_dataset()


def record_quick_dataset():
    """Record a quick dataset for testing."""
    import config
    # Temporarily set to 3 recordings per digit
    original_value = config.NUM_RECORDINGS_PER_DIGIT
    config.NUM_RECORDINGS_PER_DIGIT = 3
    
    from record_dataset import record_dataset, NUM_RECORDINGS_PER_DIGIT
    
    # Override the module-level variable too
    import record_dataset as rd
    rd.NUM_RECORDINGS_PER_DIGIT = 3
    
    record_dataset()
    
    # Restore original value
    config.NUM_RECORDINGS_PER_DIGIT = original_value


def train_model():
    """Train the CNN model."""
    from train_model import train_model as do_training
    do_training()


def start_realtime_recognition():
    """Start real-time recognition."""
    from config import MODEL_PATH
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please train the model first (option 4)")
        return
    
    from realtime_recognition import RealTimeRecognizer
    recognizer = RealTimeRecognizer()
    recognizer.start()


def test_single_recognition():
    """Test single word recognition."""
    from config import MODEL_PATH
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please train the model first (option 4)")
        return
    
    from realtime_recognition import test_recognition
    test_recognition()


def show_system_info():
    """Display system information."""
    import sounddevice as sd
    import tensorflow as tf
    from config import (SAMPLE_RATE, DURATION, N_MFCC, DIGITS, 
                       MODEL_PATH, DATASET_DIR)
    
    print("\n=== System Information ===\n")
    
    # Python & Libraries
    print("Software:")
    print(f"  Python: {sys.version}")
    print(f"  TensorFlow: {tf.__version__}")
    
    # Audio
    print("\nAudio Settings:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Recording Duration: {DURATION} seconds")
    print(f"  MFCC Coefficients: {N_MFCC}")
    
    # Dataset
    print("\nDataset:")
    print(f"  Directory: {DATASET_DIR}")
    print(f"  Classes: {', '.join(DIGITS)}")
    
    if os.path.exists(DATASET_DIR):
        total_files = 0
        for digit in DIGITS:
            digit_dir = os.path.join(DATASET_DIR, digit)
            if os.path.exists(digit_dir):
                files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
                total_files += len(files)
                print(f"    {digit}: {len(files)} files")
        print(f"  Total recordings: {total_files}")
    else:
        print("  (Dataset not created yet)")
    
    # Model
    print("\nModel:")
    print(f"  Path: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        print("  Status: Trained ✓")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Parameters: {model.count_params():,}")
    else:
        print("  Status: Not trained yet")
    
    # Audio Devices
    print("\nAudio Devices:")
    default_input = sd.query_devices(kind='input')
    print(f"  Default Input: {default_input['name']}")
    

def main():
    """Main function."""
    print_header()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    
    while True:
        print_menu()
        choice = input("Select option: ").strip().lower()
        
        try:
            if choice == '1':
                test_microphone()
            elif choice == '2':
                record_full_dataset()
            elif choice == '3':
                record_quick_dataset()
            elif choice == '4':
                train_model()
            elif choice == '5':
                start_realtime_recognition()
            elif choice == '6':
                test_single_recognition()
            elif choice == '7':
                show_system_info()
            elif choice == 'q':
                print("\nGoodbye!")
                break
            else:
                print("Invalid option. Please try again.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
