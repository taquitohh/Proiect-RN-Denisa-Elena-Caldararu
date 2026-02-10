"""Training pipeline for the neural network model (Etapa 5.2).

This script performs training with basic regularization features (early stopping,
learning rate scheduling) and light tabular augmentation.
It loads the final preprocessed datasets, builds the MLP model defined in
src/neural_network/model.py, compiles it, trains it, and saves both the trained
model and training history.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.neural_network.model import build_model


DATA_DIR = Path("data") / "chairs"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def load_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and validation datasets from CSV files."""
    x_train = pd.read_csv(DATA_DIR / "train" / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "train" / "y_train.csv").squeeze()
    x_val = pd.read_csv(DATA_DIR / "validation" / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "validation" / "y_val.csv").squeeze()
    return x_train, y_train, x_val, y_val


def augment_tabular(x_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply light Gaussian noise to continuous features only."""
    continuous_cols = [
        "seat_height",
        "seat_width",
        "seat_depth",
        "leg_thickness",
        "backrest_height",
    ]

    x_aug = x_train.copy()
    std = x_train[continuous_cols].std().fillna(0.0)
    noise = np.random.normal(loc=0.0, scale=0.02 * std.values, size=x_aug[continuous_cols].shape)
    x_aug[continuous_cols] = x_aug[continuous_cols] + noise

    if "has_backrest" in x_aug.columns:
        x_aug.loc[x_aug["has_backrest"] == 0, "backrest_height"] = 0.0

    x_out = pd.concat([x_train, x_aug], ignore_index=True)
    y_out = pd.concat([y_train, y_train], ignore_index=True)
    return x_out, y_out


def train(epochs: int = 10, batch_size: int = 32) -> None:
    """Train the model with baseline hyperparameters and save outputs."""
    x_train, y_train, x_val, y_val = load_datasets()
    x_train, y_train = augment_tabular(x_train, y_train)

    input_dim = x_train.shape[1]
    num_classes = int(pd.Series(y_train).nunique())

    model = build_model(input_dim=input_dim, num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODELS_DIR / "chair_model.h5")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(RESULTS_DIR / "chair_training_history.csv", index=False)

    final_train_acc = history.history.get("accuracy", [None])[-1]
    final_val_acc = history.history.get("val_accuracy", [None])[-1]
    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    train()
