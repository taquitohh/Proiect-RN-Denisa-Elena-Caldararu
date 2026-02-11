"""Masoara latenta de inferenta pentru modelul chair (sample unic)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "chair_model.h5"
TEST_X_PATH = PROJECT_ROOT / "data" / "chairs" / "test" / "X_test.csv"


def measure_latency(runs: int = 200, warmup: int = 20) -> float:
    """Returneaza latenta medie in milisecunde pentru inferenta pe un sample."""
    x_test = pd.read_csv(TEST_X_PATH)
    sample = x_test.iloc[[0]]

    model = tf.keras.models.load_model(MODEL_PATH)

    for _ in range(warmup):
        _ = model.predict(sample, verbose=0)

    start = time.perf_counter()
    for _ in range(runs):
        _ = model.predict(sample, verbose=0)
    end = time.perf_counter()

    avg_seconds = (end - start) / runs
    return avg_seconds * 1000.0


def main() -> None:
    avg_ms = measure_latency()
    print(f"Average single-sample latency: {avg_ms:.3f} ms")


if __name__ == "__main__":
    main()
