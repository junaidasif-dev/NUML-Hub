"""
Real-Time Speech Recognition System
Listens to microphone and recognizes spoken digits in real-time
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
from collections import deque
import threading

from config import (SAMPLE_RATE, DURATION, CONFIDENCE_THRESHOLD, 
                    SILENCE_THRESHOLD, DIGITS, MODEL_PATH, N_MFCC)
from feature_extraction import (extract_mfcc_features, pad_or_truncate_features, 
                                 get_target_length, normalize_features)


class RealTimeRecognizer:
    """Real-time speech recognition system."""
    
    def __init__(self, model_path=MODEL_PATH):
        """
        Initialize the recognizer.
        
        Args:
            model_path: Path to the trained model
        """
        self.sample_rate = SAMPLE_RATE
        self.buffer_duration = DURATION
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.target_length = get_target_length()
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.buffer_size)
        
        # Load model and parameters
        self.model = None
        self.label_map = None
        self.reverse_label_map = None
        self.norm_mean = None
        self.norm_std = None
        
        # State
        self.is_running = False
        self.stream = None
        
        # Voice activity detection
        self.vad_threshold = SILENCE_THRESHOLD
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Cooldown to prevent repeated recognitions
        self.last_recognition_time = 0
        self.cooldown_period = 0.5  # seconds
        
        # Load model
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load the trained model and normalization parameters."""
        print("Loading model...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load normalization parameters
        norm_path = 'models/normalization_params.npy'
        if os.path.exists(norm_path):
            norm_params = np.load(norm_path, allow_pickle=True).item()
            self.norm_mean = norm_params['mean']
            self.norm_std = norm_params['std']
            print(f"Normalization params loaded: mean={self.norm_mean:.4f}, std={self.norm_std:.4f}")
        else:
            print("Warning: Normalization parameters not found. Using default values.")
            self.norm_mean = 0
            self.norm_std = 1
        
        # Load label map
        label_map_path = 'models/label_map.npy'
        if os.path.exists(label_map_path):
            self.label_map = np.load(label_map_path, allow_pickle=True).item()
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            print(f"Labels: {list(self.label_map.keys())}")
        else:
            # Default to digits
            self.label_map = {digit: idx for idx, digit in enumerate(DIGITS)}
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Update buffer (shift left and add new samples)
        audio_data = indata[:, 0].astype(np.float32)
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data):] = audio_data
    
    def _compute_rms(self, audio):
        """Compute RMS (Root Mean Square) energy of audio."""
        return np.sqrt(np.mean(audio**2))
    
    def _is_speech(self, audio):
        """Simple Voice Activity Detection using RMS energy."""
        rms = self._compute_rms(audio)
        return rms > self.vad_threshold
    
    def _preprocess_audio(self, audio):
        """Preprocess audio for model inference."""
        # Extract MFCC features
        features = extract_mfcc_features(audio, self.sample_rate)
        
        # Pad or truncate
        features = pad_or_truncate_features(features, self.target_length)
        
        # Normalize
        features = (features - self.norm_mean) / (self.norm_std + 1e-8)
        
        # Reshape for model: (1, height, width, channels)
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        
        return features
    
    def _predict(self, audio):
        """
        Make prediction on audio segment.
        
        Returns:
            (predicted_label, confidence) or (None, 0) if no speech detected
        """
        # Check for speech activity
        if not self._is_speech(audio):
            return None, 0
        
        # Preprocess
        features = self._preprocess_audio(audio)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get label
        predicted_label = self.reverse_label_map.get(predicted_class, "unknown")
        
        return predicted_label, confidence
    
    def _recognition_loop(self):
        """Main recognition loop."""
        print("\n" + "="*50)
        print("REAL-TIME SPEECH RECOGNITION ACTIVE")
        print("="*50)
        print(f"Listening for digits: {', '.join(DIGITS)}")
        print(f"Confidence threshold: {self.confidence_threshold*100:.0f}%")
        print("Press Ctrl+C to stop\n")
        
        while self.is_running:
            try:
                # Get current buffer
                audio = self.audio_buffer.copy()
                
                # Predict
                label, confidence = self._predict(audio)
                
                # Check cooldown
                current_time = time.time()
                if current_time - self.last_recognition_time < self.cooldown_period:
                    time.sleep(0.05)
                    continue
                
                # Display result
                if label is not None:
                    if confidence >= self.confidence_threshold:
                        print(f"Recognized: {label.upper()} (Confidence: {confidence*100:.2f}%)")
                        self.last_recognition_time = current_time
                    else:
                        # Low confidence - show but mark as uncertain
                        rms = self._compute_rms(audio)
                        if rms > self.vad_threshold * 2:  # Only show if clearly speaking
                            print(f"[Low confidence] {label} ({confidence*100:.2f}%)")
                            self.last_recognition_time = current_time
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in recognition loop: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start real-time recognition."""
        if self.is_running:
            print("Recognition is already running!")
            return
        
        print("Starting audio stream...")
        
        # Create audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            callback=self._audio_callback
        )
        
        self.is_running = True
        self.stream.start()
        
        # Run recognition loop
        try:
            self._recognition_loop()
        except KeyboardInterrupt:
            print("\n\nStopping recognition...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop real-time recognition."""
        self.is_running = False
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        print("Recognition stopped.")
    
    def recognize_once(self, duration=1.0):
        """
        Record and recognize a single utterance.
        
        Args:
            duration: Recording duration in seconds
        
        Returns:
            (label, confidence)
        """
        print(f"Recording for {duration} seconds...")
        
        # Record audio
        audio = sd.rec(int(duration * self.sample_rate),
                       samplerate=self.sample_rate,
                       channels=1,
                       dtype='float32')
        sd.wait()
        audio = audio.flatten()
        
        print("Processing...")
        
        # Predict
        label, confidence = self._predict(audio)
        
        if label is not None:
            print(f"Result: {label.upper()} (Confidence: {confidence*100:.2f}%)")
        else:
            print("No speech detected.")
        
        return label, confidence


def test_recognition():
    """Test the recognition system with single recordings."""
    print("\n=== Speech Recognition Test Mode ===")
    
    try:
        recognizer = RealTimeRecognizer()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train_model.py")
        return
    
    while True:
        print("\nOptions:")
        print("  [1] Record and recognize single word")
        print("  [2] Start continuous recognition")
        print("  [q] Quit")
        
        choice = input("Select option: ").strip().lower()
        
        if choice == '1':
            input("Press Enter and then speak a digit...")
            recognizer.recognize_once()
        elif choice == '2':
            recognizer.start()
        elif choice == 'q':
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)
    
    # Run test mode or continuous mode based on argument
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        recognizer = RealTimeRecognizer()
        recognizer.start()
    else:
        test_recognition()
