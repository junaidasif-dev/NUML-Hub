"""
Feature Extraction Module
Extracts MFCC features with delta and delta-delta coefficients
"""

import os
import numpy as np
import librosa
from config import (SAMPLE_RATE, DURATION, N_MFCC, N_FFT, HOP_LENGTH, 
                    N_MELS, DIGITS, DATASET_DIR)


def extract_mfcc_features(audio, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, 
                          n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Extract MFCC features along with delta and delta-delta coefficients.
    
    Args:
        audio: Audio signal (numpy array)
        sample_rate: Sampling rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
    
    Returns:
        Stacked MFCC features (n_mfcc * 3, time_frames)
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc,
                                   n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Compute delta (first derivative)
    delta_mfccs = librosa.feature.delta(mfccs)
    
    # Compute delta-delta (second derivative)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Stack all features
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    return features


def pad_or_truncate_features(features, target_length):
    """
    Pad or truncate features to a fixed time length.
    
    Args:
        features: Feature matrix (n_features, time_frames)
        target_length: Target number of time frames
    
    Returns:
        Padded/truncated features
    """
    current_length = features.shape[1]
    
    if current_length < target_length:
        # Pad with zeros
        padding = np.zeros((features.shape[0], target_length - current_length))
        features = np.hstack([features, padding])
    elif current_length > target_length:
        # Truncate
        features = features[:, :target_length]
    
    return features


def get_target_length(duration=DURATION, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """Calculate the target number of time frames for given duration."""
    n_samples = int(duration * sample_rate)
    return int(np.ceil(n_samples / hop_length)) + 1


def load_and_process_audio(filepath, duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Load audio file and process it for feature extraction.
    
    Args:
        filepath: Path to audio file
        duration: Target duration in seconds
        sample_rate: Target sample rate
    
    Returns:
        Processed audio signal
    """
    # Load audio file
    audio, sr = librosa.load(filepath, sr=sample_rate, duration=duration)
    
    # Ensure consistent length
    target_samples = int(duration * sample_rate)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    
    return audio


def extract_features_from_file(filepath, target_length=None):
    """
    Extract features from a single audio file.
    
    Args:
        filepath: Path to audio file
        target_length: Target time frames (calculated if None)
    
    Returns:
        Feature matrix (n_features, target_length)
    """
    if target_length is None:
        target_length = get_target_length()
    
    # Load and process audio
    audio = load_and_process_audio(filepath)
    
    # Extract MFCC features
    features = extract_mfcc_features(audio)
    
    # Pad or truncate to target length
    features = pad_or_truncate_features(features, target_length)
    
    return features


def prepare_dataset(dataset_dir=DATASET_DIR, digits=DIGITS):
    """
    Prepare the complete dataset for training.
    
    Args:
        dataset_dir: Directory containing digit subdirectories
        digits: List of digit labels
    
    Returns:
        X: Feature array (n_samples, n_features, time_frames, 1)
        y: Label array (n_samples,)
        label_map: Dictionary mapping digit names to indices
    """
    X = []
    y = []
    label_map = {digit: idx for idx, digit in enumerate(digits)}
    
    target_length = get_target_length()
    print(f"Target time frames: {target_length}")
    
    for digit in digits:
        digit_dir = os.path.join(dataset_dir, digit)
        
        if not os.path.exists(digit_dir):
            print(f"Warning: Directory not found - {digit_dir}")
            continue
        
        wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
        print(f"Processing '{digit}': {len(wav_files)} files")
        
        for wav_file in wav_files:
            filepath = os.path.join(digit_dir, wav_file)
            try:
                features = extract_features_from_file(filepath, target_length)
                X.append(features)
                y.append(label_map[digit])
            except Exception as e:
                print(f"  Error processing {wav_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN input: (samples, height, width, channels)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    print(f"\nDataset prepared:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Number of classes: {len(label_map)}")
    
    return X, y, label_map


def normalize_features(X, mean=None, std=None):
    """
    Normalize features using z-score normalization.
    
    Args:
        X: Feature array
        mean: Precomputed mean (for inference)
        std: Precomputed std (for inference)
    
    Returns:
        Normalized features, mean, std
    """
    if mean is None:
        mean = np.mean(X)
    if std is None:
        std = np.std(X)
    
    X_normalized = (X - mean) / (std + 1e-8)
    return X_normalized, mean, std


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction...")
    
    # Check if dataset exists
    if os.path.exists(DATASET_DIR):
        X, y, label_map = prepare_dataset()
        X_norm, mean, std = normalize_features(X)
        print(f"\nNormalized X shape: {X_norm.shape}")
        print(f"Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"Label map: {label_map}")
    else:
        print(f"Dataset directory '{DATASET_DIR}' not found.")
        print("Please run record_dataset.py first to create recordings.")
