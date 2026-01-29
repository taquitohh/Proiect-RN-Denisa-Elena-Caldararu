"""Evaluate trained RN model on the test set (Etapa 5.3).

This script loads the trained model and the test dataset, runs inference,
computes Accuracy and F1-score (macro), prints the metrics, and saves them
in results/test_metrics.json. It optionally saves a confusion matrix plot to
`docs/confusion_matrix.png` if matplotlib is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


DATA_DIR = Path("data")
MODEL_PATH = Path("models") / "trained_model.h5"
RESULTS_PATH = Path("results") / "test_metrics.json"
CONFUSION_PATH = Path("docs") / "confusion_matrix.png"


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load test dataset from CSV files."""
    x_test = pd.read_csv(DATA_DIR / "test" / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "test" / "y_test.csv").squeeze()
    return x_test, y_test


def compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute macro-averaged F1-score without external dependencies."""
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        if tp == 0 and (fp > 0 or fn > 0):
            f1_scores.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            f1_scores.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def save_confusion_matrix(cm: np.ndarray) -> None:
    """Save confusion matrix plot if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping confusion matrix plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)


def evaluate() -> None:
    """Run inference on test set and compute evaluation metrics."""
    x_test, y_test = load_test_data()

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    y_true = y_test.to_numpy()
    accuracy = float(np.mean(y_pred == y_true))

    num_classes = probabilities.shape[1]
    f1_macro = compute_f1_macro(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test F1-score (macro): {f1_macro:.2f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {"accuracy": accuracy, "f1_macro": f1_macro},
            file,
            indent=2,
        )

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    save_confusion_matrix(cm)


if __name__ == "__main__":
    evaluate()
