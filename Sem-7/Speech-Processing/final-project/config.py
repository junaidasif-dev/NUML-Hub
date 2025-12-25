"""
Configuration settings for Speech Recognition System
"""

# Audio Settings
SAMPLE_RATE = 16000  # 16 kHz sampling rate
DURATION = 1.0       # 1 second recording duration
CHANNELS = 1         # Mono audio

# MFCC Feature Settings
N_MFCC = 13          # Number of MFCC coefficients
N_FFT = 512          # FFT window size
HOP_LENGTH = 160     # Hop length (10ms at 16kHz)
N_MELS = 40          # Number of mel bands

# Dataset Settings
DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
NUM_RECORDINGS_PER_DIGIT = 10  # Number of recordings per digit
DATASET_DIR = 'dataset'
MODEL_PATH = 'models/speech_recognition_model.keras'

# Training Settings
BATCH_SIZE = 4
EPOCHS = 5
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Real-time Recognition Settings
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to recognize a command
SILENCE_THRESHOLD = 0.01    # RMS threshold for voice activity detection
BUFFER_DURATION = 1.0       # Audio buffer duration in seconds
