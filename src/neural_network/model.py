"""Definirea modelului pentru clasificarea scaunelor (Etapa 4).

Modulul expune o singura functie, `build_model`, care returneaza
un MLP Keras potrivit pentru inputuri tabulare numerice.
"""

from __future__ import annotations

import tensorflow as tf


def build_model(input_dim: int = 8, num_classes: int = 4) -> tf.keras.Model:
    """Creeaza un MLP pentru clasificare multi-clasa.

    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.

    Returns:
        Un model Keras.
    """
    # MLP simplu pentru clasificare tabulara cu doua straturi ascunse.
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
