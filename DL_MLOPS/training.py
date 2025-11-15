# training.py
"""
Train CNN model with MLFlow experiment tracking.
Handles class imbalance, early stopping, and model versioning.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
import mlflow
import mlflow.keras

from model_cnn import create_sequences, create_labels, build_cnn_model


# =========================================================
# TRAINING SCRIPT
# =========================================================

print("=" * 70)
print("STEP 4: CNN TRAINING WITH MLFLOW")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load preprocessed data
# ---------------------------------------------------------
print("\n[1/7] Loading data...")
train = pd.read_csv("data/processed/train_features.csv", parse_dates=["date"])
val = pd.read_csv("data/processed/val_features.csv", parse_dates=["date"])

print(f"  Train set: {len(train)} rows")
print(f"  Val set:   {len(val)} rows")

# ---------------------------------------------------------
# 2. Prepare features
# ---------------------------------------------------------
print("\n[2/7] Preparing features...")
exclude_cols = ["date", "open", "high", "low", "close", "volume", "next_return"]
features = [col for col in train.columns if col not in exclude_cols]
print(f"  Using {len(features)} features")

# ---------------------------------------------------------
# 3. Normalize data
# ---------------------------------------------------------
print("\n[3/7] Normalizing data...")
scaler = StandardScaler()
scaler.fit(train[features])

train_scaled = train.copy()
val_scaled = val.copy()
train_scaled[features] = scaler.transform(train[features])
val_scaled[features] = scaler.transform(val[features])

# Save scaler for production
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("   Scaler saved to models/scaler.pkl")

# ---------------------------------------------------------
# 4. Create sequences and labels
# ---------------------------------------------------------
print("\n[4/7] Creating sequences and labels...")
WINDOW_SIZE = 30

X_train, idx_train = create_sequences(train_scaled[features].values, WINDOW_SIZE)
X_val, idx_val = create_sequences(val_scaled[features].values, WINDOW_SIZE)

y_train = create_labels(train_scaled, idx_train)
y_val = create_labels(val_scaled, idx_val)

# ---------------------------------------------------------
# 5. Build CNN model
# ---------------------------------------------------------
print("\n[5/7] Building CNN model...")
model = build_cnn_model(window_size=WINDOW_SIZE, n_features=len(features))

# Handle class imbalance with class weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {int(i): float(w) for i, w in zip(classes, weights)}
print(f"  Class weights: {class_weights}")

# Define early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ---------------------------------------------------------
# 6. Train with MLFlow tracking
# ---------------------------------------------------------
print("\n[6/7] Training with MLFlow tracking...")
print("=" * 70)

mlflow.set_experiment("trading_cnn")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("window_size", WINDOW_SIZE)
    mlflow.log_param("n_features", len(features))
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("architecture", "CNN_2layers")

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Log metrics
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("val_loss", val_loss)

    # Save and log model
    model.save("models/cnn_model.keras")
    mlflow.keras.log_model(model, artifact_path="cnn_model")
    print("\n✅ Local model saved to models/cnn_model.keras")


# ---------------------------------------------------------
# 7. Display final results
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING RESULTS SUMMARY")
print("=" * 70)
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Val Accuracy:   {val_acc:.2%}")
print(f"Train Loss:     {train_loss:.4f}")
print(f"Val Loss:       {val_loss:.4f}")
print("\n To visualize experiments in MLFlow:")
print("   > mlflow ui")
print("   Open http://localhost:5000")
print("=" * 70)
