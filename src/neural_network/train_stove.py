"""Pipeline de antrenare pentru modelul de plite."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import tensorflow as tf

from src.neural_network.model import build_model


# Cai pentru dataset si artefacte (model/rezultate).
DATA_DIR = Path("data") / "stoves"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_PATH = MODELS_DIR / "stove_model.h5"
HISTORY_PATH = RESULTS_DIR / "stove_training_history.csv"
METRICS_PATH = RESULTS_DIR / "stove_training_metrics.json"


def load_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Incarca dataset-urile train si validation din CSV."""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

    # Incarca features si label-uri pentru train/validation.
    x_train = pd.read_csv(DATA_DIR / "train" / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "train" / "y_train.csv").squeeze()
    x_val = pd.read_csv(DATA_DIR / "validation" / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "validation" / "y_val.csv").squeeze()
    return x_train, y_train, x_val, y_val


def train(epochs: int = 10, batch_size: int = 32) -> None:
    """Antreneaza modelul de plite si salveaza output-urile."""
    # Pregateste datele de antrenare si validare.
    x_train, y_train, x_val, y_val = load_datasets()

    input_dim = x_train.shape[1]
    num_classes = int(pd.Series(y_train).nunique())

    # Construieste si compileaza modelul MLP.
    model = build_model(input_dim=input_dim, num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Antreneaza modelul si retine istoricul.
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Salveaza modelul si artefactele de antrenare.
    model.save(MODEL_PATH)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH, index=False)

    # Rezumat metrici finale pentru verificare rapida.
    metrics = {
        "train_accuracy": history.history.get("accuracy", [None])[-1],
        "train_loss": history.history.get("loss", [None])[-1],
        "val_accuracy": history.history.get("val_accuracy", [None])[-1],
        "val_loss": history.history.get("val_loss", [None])[-1],
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    print(f"Final train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Final validation accuracy: {metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    train()
