"""Exporta modelul de scaune in ONNX si masoara latenta."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnxruntime as ort


# Cai pentru export ONNX si datele de benchmark.
MODEL_PATH = Path("models") / "chair_model.h5"
ONNX_PATH = Path("models") / "chair_model.onnx"
DATA_PATH = Path("data") / "chairs" / "test" / "X_test.csv"


def export_model() -> None:
    """Exporta modelul Keras in format ONNX."""
    # Incarca modelul Keras si asigura iesirea definita.
    model = tf.keras.models.load_model(MODEL_PATH)
    input_dim = model.input_shape[1]
    model(np.zeros((1, input_dim), dtype=np.float32))
    if not hasattr(model, "output_names"):
        model.output_names = [model.outputs[0].name.split(":")[0]]
    input_spec = (
        tf.TensorSpec((None, input_dim), tf.float32, name="input"),
    )
    # Converteste modelul Keras in format ONNX.
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=13,
        output_path=str(ONNX_PATH),
    )


def benchmark_latency(runs: int = 200) -> float:
    """Masoara latenta medie a inferentei ONNX (batch=1)."""
    # Foloseste un singur sample pentru latenta batch=1.
    x_test = pd.read_csv(DATA_PATH).astype(np.float32)
    sample = x_test.iloc[[0]].to_numpy()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Rulari de warm-up pentru stabilizarea timpului.
    for _ in range(10):
        session.run(None, {input_name: sample})

    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: sample})
    elapsed_ms = (time.perf_counter() - start) / runs * 1000.0
    return elapsed_ms


def main() -> None:
    """Ruleaza exportul si afiseaza latenta medie."""
    # Exporta ONNX si afiseaza latenta medie.
    export_model()
    avg_ms = benchmark_latency()
    print(f"ONNX avg latency (CPU, batch=1): {avg_ms:.2f} ms")


if __name__ == "__main__":
    main()
