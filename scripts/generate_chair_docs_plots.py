"""Genereaza graficele de documentatie pentru modelul chair (Etapa 6)."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
OPT_DIR = DOCS_DIR / "optimization"
RES_DIR = DOCS_DIR / "results"

MODEL_PATH = PROJECT_ROOT / "models" / "chair_model.h5"
TEST_X_PATH = PROJECT_ROOT / "data" / "chairs" / "test" / "X_test.csv"
TEST_Y_PATH = PROJECT_ROOT / "data" / "chairs" / "test" / "y_test.csv"
HISTORY_PATH = RESULTS_DIR / "chair_training_history.csv"
TEST_METRICS_PATH = RESULTS_DIR / "chair_test_metrics.json"

LABEL_NAMES = {
    0: "chair_simple",
    1: "chair_with_backrest",
    2: "bar_chair",
    3: "stool",
}


def ensure_dirs() -> None:
    """Creeaza folderele de output daca lipsesc."""
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)


def load_test_metrics() -> dict:
    """Incarca metricile de test din JSON."""
    with TEST_METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_metrics_evolution() -> None:
    """Genereaza evolutia Accuracy/F1 intre Etapele 4-6."""
    import matplotlib.pyplot as plt

    metrics = load_test_metrics()

    # Valorile Etapa 4 sunt tinte aproximative din cerinte.
    stages = ["Etapa 4", "Etapa 5", "Etapa 6"]
    accuracy = [0.70, metrics["accuracy"], metrics["accuracy"]]
    f1_macro = [0.65, metrics["f1_macro"], metrics["f1_macro"]]

    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#3b82f6")
    ax.bar(x + width / 2, f1_macro, width, label="F1 macro", color="#10b981")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Score")
    ax.set_title("Metrici - evolutie Etapa 4 -> 6 (chair)")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(RES_DIR / "metrics_evolution.png", dpi=150)
    plt.close(fig)


def plot_learning_curves() -> None:
    """Genereaza curbele de invatare train/val pentru modelul final."""
    import matplotlib.pyplot as plt

    history = pd.read_csv(HISTORY_PATH)
    epochs = np.arange(1, len(history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["accuracy"], label="train", color="#3b82f6")
    axes[0].plot(epochs, history["val_accuracy"], label="val", color="#10b981")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], label="train", color="#f97316")
    axes[1].plot(epochs, history["val_loss"], label="val", color="#ef4444")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle("Learning Curves - model optimizat (chair)")
    fig.tight_layout()
    fig.savefig(RES_DIR / "learning_curves_final.png", dpi=150)
    plt.close(fig)


def plot_optimization_comparisons() -> None:
    """Genereaza comparatia Accuracy si F1 intre experimente."""
    import matplotlib.pyplot as plt

    experiments = ["Baseline", "Exp 1", "Exp 2", "Exp 3", "Exp 4"]
    accuracy = [0.9911, 0.9867, 0.9893, 0.9862, 0.9858]
    f1_macro = [0.9915, 0.9861, 0.9894, 0.9865, 0.9849]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(experiments, accuracy, color="#3b82f6")
    ax.set_ylim(0.97, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy comparison - chair")
    fig.tight_layout()
    fig.savefig(OPT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(experiments, f1_macro, color="#10b981")
    ax.set_ylim(0.97, 1.0)
    ax.set_ylabel("F1 macro")
    ax.set_title("F1 comparison - chair")
    fig.tight_layout()
    fig.savefig(OPT_DIR / "f1_comparison.png", dpi=150)
    plt.close(fig)


def plot_example_predictions() -> None:
    """Randeaza un tabel cu predictii exemplu din setul de test."""
    import matplotlib.pyplot as plt

    x_test = pd.read_csv(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH).squeeze().to_numpy()

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    rng = np.random.default_rng(42)
    incorrect = np.where(y_pred != y_test)[0]
    correct = np.where(y_pred == y_test)[0]

    sample_indices = []
    if len(incorrect) > 0:
        take_incorrect = min(6, len(incorrect))
        sample_indices.extend(rng.choice(incorrect, size=take_incorrect, replace=False).tolist())
    if len(sample_indices) < 12:
        remaining = 12 - len(sample_indices)
        sample_indices.extend(rng.choice(correct, size=remaining, replace=False).tolist())

    rows = []
    for idx in sample_indices:
        rows.append(
            [
                int(idx),
                LABEL_NAMES.get(int(y_test[idx]), str(int(y_test[idx]))),
                LABEL_NAMES.get(int(y_pred[idx]), str(int(y_pred[idx]))),
                f"{confidence[idx]:.3f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Index", "True", "Pred", "Conf"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    ax.set_title("Example predictions (chair)")
    fig.tight_layout()
    fig.savefig(RES_DIR / "example_predictions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Genereaza toate graficele cerute."""
    ensure_dirs()
    plot_metrics_evolution()
    plot_learning_curves()
    plot_optimization_comparisons()
    plot_example_predictions()


if __name__ == "__main__":
    main()
