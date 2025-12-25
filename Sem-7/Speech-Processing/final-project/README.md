# Real-Time Speech Recognition System

A Python-based system that recognizes spoken digits (zero to nine) in real-time from a live microphone audio stream.

## Project Structure

```
sp/
├── venv/                    # Virtual environment
├── dataset/                 # Recorded audio samples (created by recording)
│   ├── zero/
│   ├── one/
│   ├── ...
│   └── nine/
├── models/                  # Trained models and parameters
│   ├── speech_recognition_model.keras
│   ├── normalization_params.npy
│   └── label_map.npy
├── config.py                # Configuration settings
├── record_dataset.py        # Dataset recording script
├── feature_extraction.py    # MFCC feature extraction
├── train_model.py           # CNN model training
├── realtime_recognition.py  # Real-time recognition system
├── main.py                  # Main runner with menu interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup Instructions

### 1. Activate Virtual Environment

```powershell
# Windows
cd c:\Users\junai\OneDrive\Desktop\sp
.\venv\Scripts\activate
```

### 2. Verify Installation

```powershell
python -c "import tensorflow; import librosa; import sounddevice; print('All packages installed!')"
```

## Usage Guide

### Step 1: Run the Main Program

```powershell
python main.py
```

This opens a menu with all options:

1. **Test Microphone** - Verify your microphone is working
2. **Record Dataset** - Record 10 samples per digit (recommended)
3. **Quick Record** - Record 3 samples per digit (for testing)
4. **Train Model** - Train the CNN on your recordings
5. **Start Real-Time Recognition** - Begin live recognition
6. **Test Single Recognition** - Test with individual recordings
7. **View System Info** - Check system configuration

### Step 2: Test Your Microphone (Option 1)

Before recording, test that your microphone is working properly.

### Step 3: Record Dataset (Option 2 or 3)

- **Full Dataset (Option 2)**: Records 10 samples per digit (100 total recordings)
- **Quick Dataset (Option 3)**: Records 3 samples per digit (30 total recordings)

**Recording Tips:**
- Use a quiet environment
- Speak clearly and consistently
- Keep the same distance from microphone
- Wait for the countdown before speaking

### Step 4: Train the Model (Option 4)

After recording your dataset, train the CNN model:
- Takes a few minutes depending on dataset size
- Saves the best model automatically
- Displays training accuracy and loss graphs

### Step 5: Start Real-Time Recognition (Option 5)

Once trained, start recognizing spoken digits in real-time:
- Speak a digit clearly
- See the recognized result with confidence percentage
- Press Ctrl+C to stop

## Individual Script Usage

### Record Dataset Only
```powershell
python record_dataset.py
```

### Extract Features Only
```powershell
python feature_extraction.py
```

### Train Model Only
```powershell
python train_model.py
```

### Real-Time Recognition Only
```powershell
# Interactive mode
python realtime_recognition.py

# Continuous mode
python realtime_recognition.py --continuous
```

## Configuration

Edit `config.py` to customize:

```python
# Audio Settings
SAMPLE_RATE = 16000      # Sampling rate (Hz)
DURATION = 1.0           # Recording duration (seconds)

# MFCC Settings
N_MFCC = 13              # Number of MFCC coefficients

# Training Settings
EPOCHS = 50              # Training epochs
BATCH_SIZE = 32          # Batch size

# Recognition Settings
CONFIDENCE_THRESHOLD = 0.6   # Minimum confidence (60%)
SILENCE_THRESHOLD = 0.01     # Voice activity threshold
```

## Technical Details

### Feature Extraction
- **MFCCs**: 13 Mel-Frequency Cepstral Coefficients
- **Delta MFCCs**: First derivative of MFCCs
- **Delta-Delta MFCCs**: Second derivative of MFCCs
- **Total Features**: 39 (13 × 3)

### CNN Architecture
```
Input → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout
      → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout
      → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout
      → Conv2D(256) → BatchNorm → ReLU
      → GlobalAveragePooling
      → Dense(128) → BatchNorm → ReLU → Dropout
      → Dense(64) → BatchNorm → ReLU → Dropout
      → Dense(10) → Softmax
```

### Voice Activity Detection (VAD)
- RMS energy-based detection
- Prevents recognition on silence
- Configurable threshold

## Troubleshooting

### "No module named..." Error
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Microphone Not Detected
- Check Windows sound settings
- Verify microphone permissions
- Try a different USB port

### Low Recognition Accuracy
- Record more samples (aim for 10+ per digit)
- Ensure consistent speaking style
- Reduce background noise
- Increase training epochs

### Model Not Found Error
Run training first: `python train_model.py`

## Expected Output

### During Recognition
```
==================================================
REAL-TIME SPEECH RECOGNITION ACTIVE
==================================================
Listening for digits: zero, one, two, three, four, five, six, seven, eight, nine
Confidence threshold: 60%
Press Ctrl+C to stop

Recognized: FIVE (Confidence: 94.32%)
Recognized: THREE (Confidence: 87.65%)
[Low confidence] seven (42.18%)
Recognized: ZERO (Confidence: 98.50%)
```

## License

This project is for educational purposes.
