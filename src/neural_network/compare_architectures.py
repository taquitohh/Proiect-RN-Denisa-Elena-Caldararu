"""Compara mai multe arhitecturi MLP pe dataset-ul de scaune (Etapa 5/6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# Radacina pentru dataset-urile chair.
DATA_DIR = Path("data") / "chairs"


@dataclass
class ExperimentResult:
    name: str
    params: int
    accuracy: float
    f1_macro: float
    train_seconds: float


def load_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Incarca spliturile train/validation/test din CSV."""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

    # Incarca spliturile train/validation/test.
    x_train = pd.read_csv(DATA_DIR / "train" / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "train" / "y_train.csv").squeeze()
    x_val = pd.read_csv(DATA_DIR / "validation" / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "validation" / "y_val.csv").squeeze()
    x_test = pd.read_csv(DATA_DIR / "test" / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "test" / "y_test.csv").squeeze()
    return x_train, y_train, x_val, y_val, x_test, y_test


def augment_tabular(x_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Aplica zgomot gaussian usor doar pe coloane continue."""
    # Adauga zgomot gaussian doar pe feature-urile continue.
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


def build_variant(name: str, input_dim: int, num_classes: int) -> tf.keras.Model:
    """Construieste un model pentru varianta ceruta."""
    # Mapare intre numele variantei si dimensiunile straturilor ascunse.
    if name == "baseline_32_16":
        hidden = [32, 16]
    elif name == "narrow_16_8":
        hidden = [16, 8]
    elif name == "wider_64_32":
        hidden = [64, 32]
    elif name == "deeper_64_32_16":
        hidden = [64, 32, 16]
    else:
        raise ValueError(f"Unknown variant: {name}")

    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in hidden:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return tf.keras.Sequential(layers)


def compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Calculeaza F1 macro pentru clasificare multi-clasa."""
    # Calculeaza F1 pe clasa si media macro.
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
        f1_scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def run_experiment(name: str) -> ExperimentResult:
    """Ruleaza un experiment complet pentru o varianta de arhitectura."""
    # Antreneaza si evalueaza o varianta de arhitectura cap-coada.
    x_train, y_train, x_val, y_val, x_test, y_test = load_datasets()
    x_train, y_train = augment_tabular(x_train, y_train)

    input_dim = x_train.shape[1]
    num_classes = int(pd.Series(y_train).nunique())

    model = build_variant(name, input_dim, num_classes)
    # Compileaza cu acelasi optimizer si loss pentru comparatie corecta.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Foloseste early stopping si scheduler pentru stabilitate.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    # Masoara timpul de antrenare pentru comparatie.
    start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=0,
        callbacks=callbacks,
    )
    train_seconds = time.perf_counter() - start

    # Prezice pe test pentru a calcula metricile finale.
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = y_test.to_numpy()

    accuracy = float(np.mean(y_pred == y_true))
    f1_macro = compute_f1_macro(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    params = model.count_params()
    return ExperimentResult(
        name=name,
        params=params,
        accuracy=accuracy,
        f1_macro=f1_macro,
        train_seconds=train_seconds,
    )


def main() -> None:
    """Ruleaza toate variantele si afiseaza metricile."""
    # Ruleaza toate variantele si afiseaza metricile.
    variants = ["baseline_32_16", "narrow_16_8", "wider_64_32", "deeper_64_32_16"]
    results: List[ExperimentResult] = [run_experiment(name) for name in variants]
    for res in results:
        print(
            f"{res.name}: params={res.params}, accuracy={res.accuracy:.4f}, "
            f"f1_macro={res.f1_macro:.4f}, train_s={res.train_seconds:.2f}"
        )


if __name__ == "__main__":
    main()
