"""
Dataset Recording Script
Records audio samples for each digit (zero to nine)
"""

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from config import SAMPLE_RATE, DURATION, CHANNELS, DIGITS, NUM_RECORDINGS_PER_DIGIT, DATASET_DIR


def create_dataset_directories():
    """Create directories for each digit."""
    for digit in DIGITS:
        digit_dir = os.path.join(DATASET_DIR, digit)
        os.makedirs(digit_dir, exist_ok=True)
    print(f"Created dataset directories in '{DATASET_DIR}'")


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """Record audio from microphone."""
    print("Recording...", end=" ", flush=True)
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=CHANNELS, 
                   dtype='float32')
    sd.wait()  # Wait for recording to complete
    print("Done!")
    return audio.flatten()


def save_audio(audio, filepath, sample_rate=SAMPLE_RATE):
    """Save audio to WAV file."""
    sf.write(filepath, audio, sample_rate)
    print(f"Saved: {filepath}")


def play_audio(audio, sample_rate=SAMPLE_RATE):
    """Play back the recorded audio."""
    sd.play(audio, sample_rate)
    sd.wait()


def record_dataset():
    """Main function to record the entire dataset."""
    create_dataset_directories()
    
    print("\n" + "="*60)
    print("SPEECH RECOGNITION DATASET RECORDER")
    print("="*60)
    print(f"\nYou will record each digit {NUM_RECORDINGS_PER_DIGIT} times.")
    print(f"Each recording will be {DURATION} second(s) long.")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("\nInstructions:")
    print("1. Wait for the countdown")
    print("2. Say the digit clearly when 'Recording...' appears")
    print("3. Review and re-record if needed")
    print("\n" + "="*60)
    
    input("\nPress Enter to start recording...")
    
    for digit in DIGITS:
        digit_dir = os.path.join(DATASET_DIR, digit)
        
        print(f"\n{'='*40}")
        print(f"Now recording: '{digit.upper()}'")
        print(f"{'='*40}")
        
        recording_num = 1
        while recording_num <= NUM_RECORDINGS_PER_DIGIT:
            print(f"\n--- Recording {recording_num}/{NUM_RECORDINGS_PER_DIGIT} for '{digit}' ---")
            
            # Countdown
            for i in range(3, 0, -1):
                print(f"Get ready... {i}")
                time.sleep(1)
            
            # Record
            audio = record_audio()
            
            # Options
            while True:
                print("\nOptions:")
                print("  [p] Play back the recording")
                print("  [s] Save and continue")
                print("  [r] Re-record")
                print("  [q] Quit recording session")
                
                choice = input("Your choice: ").lower().strip()
                
                if choice == 'p':
                    print("Playing back...")
                    play_audio(audio)
                elif choice == 's':
                    filepath = os.path.join(digit_dir, f"{digit}_{recording_num}.wav")
                    save_audio(audio, filepath)
                    recording_num += 1
                    break
                elif choice == 'r':
                    print("Let's record again...")
                    break
                elif choice == 'q':
                    print("\nRecording session ended early.")
                    return
                else:
                    print("Invalid choice. Please try again.")
        
        print(f"\nCompleted all recordings for '{digit}'!")
    
    print("\n" + "="*60)
    print("DATASET RECORDING COMPLETE!")
    print(f"All recordings saved to '{DATASET_DIR}' directory.")
    print("="*60)


def check_microphone():
    """Test if microphone is working."""
    print("\n=== Microphone Test ===")
    print("Available audio devices:")
    print(sd.query_devices())
    
    print(f"\nDefault input device: {sd.query_devices(kind='input')['name']}")
    
    input("\nPress Enter to record a 2-second test...")
    
    print("\nRecording 2-second test...")
    audio = record_audio(duration=2)
    
    rms = np.sqrt(np.mean(audio**2))
    print(f"RMS Level: {rms:.4f}")
    
    if rms < 0.001:
        print("WARNING: Audio level is very low. Check your microphone!")
    else:
        print("Microphone is working!")
    
    input("\nPress Enter to play back the test recording...")
    play_audio(audio)
    print("Test complete!")


if __name__ == "__main__":
    import sys
    
    print("\n=== Dataset Recording Tool ===")
    print("1. Test microphone")
    print("2. Record full dataset")
    print("3. Quick record (3 samples per digit)")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        check_microphone()
    elif choice == '2':
        record_dataset()
    elif choice == '3':
        # Quick mode for testing
        import config
        config.NUM_RECORDINGS_PER_DIGIT = 3
        NUM_RECORDINGS_PER_DIGIT = 3
        record_dataset()
    else:
        print("Invalid option. Exiting.")
