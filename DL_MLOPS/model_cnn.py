# model_cnn.py
"""
Convolutional Neural Network (CNN) model and helper functions.
Generates sliding windows, class labels, and builds a 1D CNN for time series classification.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# Sequence and Label Generation
# =========================================================

def create_sequences(data, window_size=30):
    """
    Create sliding windows from time series data.

    Example:
        If data has 100 rows and window_size=30:
        - Sequence 0: rows 0–29
        - Sequence 1: rows 1–30
        - Sequence 2: rows 2–31
        - ...

    Args:
        data (np.ndarray): Input array of shape (n_samples, n_features)
        window_size (int): Number of past time steps in each window

    Returns:
        X (np.ndarray): Sequences of shape (n_sequences, window_size, n_features)
        valid_indices (list): Row indices in the original data corresponding to each sequence
    """
    X = []
    valid_indices = []

    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        X.append(window)
        valid_indices.append(i + window_size)

    X = np.array(X)
    print(f"   Created {len(X)} sequences with shape {X.shape[1:]}")
    return X, valid_indices


def create_labels(df, valid_indices, threshold=0.01):
    """
    Generate classification labels based on next-day returns.

    Labeling rule:
        0 = SHORT (price expected to decrease)
        1 = HOLD  (price expected to remain stable)
        2 = LONG  (price expected to increase)

    Args:
        df (pd.DataFrame): DataFrame containing 'close' column
        valid_indices (list): Indices from create_sequences
        threshold (float): Percentage threshold for class separation (default 1%)

    Returns:
        np.ndarray: Array of class labels {0, 1, 2}
    """
    df = df.copy()
    df["next_return"] = df["close"].pct_change().shift(-1)

    labels = []
    for idx in valid_indices:
        next_return = df.iloc[idx]["next_return"]

        if next_return > threshold:
            labels.append(2)  # LONG
        elif next_return < -threshold:
            labels.append(0)  # SHORT
        else:
            labels.append(1)  # HOLD

    labels = np.array(labels)

    # Print class distribution
    n_short = (labels == 0).sum()
    n_hold = (labels == 1).sum()
    n_long = (labels == 2).sum()
    total = len(labels)

    print("   Labels created:")
    print(f"      SHORT: {n_short} ({n_short/total*100:.1f}%)")
    print(f"      HOLD:  {n_hold} ({n_hold/total*100:.1f}%)")
    print(f"      LONG:  {n_long} ({n_long/total*100:.1f}%)")

    return labels


# =========================================================
# CNN Model Definition
# =========================================================

def build_cnn_model(window_size, n_features):
    """
    Build a simple 1D Convolutional Neural Network for time series classification.

    Architecture:
        - Conv1D layers: extract temporal features
        - MaxPooling: reduce dimensionality
        - Dense layers: learn feature interactions
        - Dropout: prevent overfitting
        - Softmax output: 3-class prediction (short, hold, long)

    Args:
        window_size (int): Number of time steps in each sequence
        n_features (int): Number of features per time step

    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_features)),

        # First convolutional block
        layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        # Second convolutional block
        layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        # Output layer (3 classes)
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("   CNN model compiled successfully.")
    return model


# =========================================================
# Model Loading Function
# =========================================================

def load_model(model_path):
    """
    Load a trained CNN model from disk.

    Args:
        model_path (str): Path to the .h5 model file

    Returns:
        keras.Model: Loaded model
    """
    model = keras.models.load_model(model_path)
    print(f"   Model loaded from '{model_path}'")
    return model
