"""Definirea modelului pentru clasificarea scaunelor (Etapa 4).

Modulul expune o singura functie, `build_model`, care returneaza
un MLP Keras potrivit pentru inputuri tabulare numerice.
"""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.


from __future__ import annotations

import tensorflow as tf


def build_model(input_dim: int = 8, num_classes: int = 4) -> tf.keras.Model:
    """Creeaza un MLP pentru clasificare multi-clasa.

    Argumente:
        input_dim: Numarul de feature-uri de intrare.
        num_classes: Numarul de clase la iesire.

    Returneaza:
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
