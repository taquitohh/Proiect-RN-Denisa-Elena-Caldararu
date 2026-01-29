"""Model definition for the chair type classifier (Etapa 4).

This module exposes a single factory function, `build_model`, which returns a
Keras MLP suitable for tabular, numeric inputs.
"""

from __future__ import annotations

import tensorflow as tf


def build_model(input_dim: int = 8, num_classes: int = 4) -> tf.keras.Model:
    """Create an MLP model for multi-class classification.

    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.

    Returns:
        A Keras Model instance.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
