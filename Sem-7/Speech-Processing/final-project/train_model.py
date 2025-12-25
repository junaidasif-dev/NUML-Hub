"""
CNN Model Training Script
Trains a Convolutional Neural Network for speech recognition
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from config import (DIGITS, DATASET_DIR, MODEL_PATH, BATCH_SIZE, 
                    EPOCHS, VALIDATION_SPLIT, LEARNING_RATE, N_MFCC)
from feature_extraction import prepare_dataset, normalize_features, get_target_length


def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN model for speech recognition.
    
    Architecture:
    - 3 Convolutional blocks with BatchNorm and MaxPooling
    - Global Average Pooling
    - Dense layers with Dropout
    - Softmax output
    
    Args:
        input_shape: Shape of input features (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense Layers
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history saved to {save_path}")


def train_model():
    """Main training function."""
    print("="*60)
    print("SPEECH RECOGNITION MODEL TRAINING")
    print("="*60)
    
    # Check for dataset
    if not os.path.exists(DATASET_DIR):
        print(f"\nError: Dataset directory '{DATASET_DIR}' not found!")
        print("Please run record_dataset.py first to create recordings.")
        return None
    
    # Prepare dataset
    print("\n1. Preparing dataset...")
    X, y, label_map = prepare_dataset()
    
    if len(X) == 0:
        print("Error: No audio files found in dataset!")
        return None
    
    # Normalize features
    print("\n2. Normalizing features...")
    X_normalized, mean, std = normalize_features(X)
    
    # Save normalization parameters for inference
    np.save('models/normalization_params.npy', {'mean': mean, 'std': std})
    np.save('models/label_map.npy', label_map)
    
    # Split dataset
    print("\n3. Splitting dataset...")
    
    # Check if we have enough samples for stratified split
    num_classes = len(label_map)
    test_size_samples = int(len(X_normalized) * VALIDATION_SPLIT)
    
    # Need at least 1 sample per class in test set for stratification
    if test_size_samples < num_classes:
        print(f"   Warning: Small dataset ({len(X_normalized)} samples).")
        print(f"   Using simple split without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42, stratify=None
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
        )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Create model
    print("\n4. Creating CNN model...")
    input_shape = X_train.shape[1:]  # (height, width, channels)
    num_classes = len(label_map)
    
    model = create_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Data augmentation (time shift and noise)
    print("\n5. Training model...")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callback_list,
        verbose=1
    )
    
    # Evaluate model
    print("\n6. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Plot training history
    print("\n7. Plotting training history...")
    plot_training_history(history)
    
    # Save final model
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    
    # Print classification report
    print("\n8. Per-class accuracy:")
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    for class_idx in range(num_classes):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == y_test[class_mask])
            print(f"   {reverse_label_map[class_idx]}: {class_acc*100:.2f}%")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    model = train_model()
