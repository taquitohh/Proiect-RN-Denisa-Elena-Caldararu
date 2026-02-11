"""Genereaza dataset sintetic pentru clasificarea frigiderelor."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

# Dimensiune dataset si seed pentru reproductibilitate.
NUM_SAMPLES = 12000
RANDOM_SEED = 42

FRIDGE_HEIGHT_RANGE = (1.4, 2.1)
FRIDGE_WIDTH_RANGE = (0.6, 1.0)
FRIDGE_DEPTH_RANGE = (0.55, 0.8)
DOOR_THICKNESS_RANGE = (0.02, 0.05)
HANDLE_LENGTH_RANGE = (0.15, 0.45)
FREEZER_RATIO_RANGE = (0.25, 0.45)
FREEZER_POSITION_OPTIONS = (0, 1)  # 0 = top freezer, 1 = bottom freezer
STYLE_VARIANT_OPTIONS = (0, 1, 2)

# Initializeaza RNG-urile o singura data.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_sample() -> dict[str, float | int]:
    """Genereaza un sample de frigider cu valori valide."""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

    # Genereaza valori in intervalele definite.
    fridge_height = random.uniform(*FRIDGE_HEIGHT_RANGE)
    fridge_width = random.uniform(*FRIDGE_WIDTH_RANGE)
    fridge_depth = random.uniform(*FRIDGE_DEPTH_RANGE)
    door_thickness = random.uniform(*DOOR_THICKNESS_RANGE)
    handle_length = random.uniform(*HANDLE_LENGTH_RANGE)
    freezer_ratio = random.uniform(*FREEZER_RATIO_RANGE)
    freezer_position = random.choice(FREEZER_POSITION_OPTIONS)
    style_variant = random.choice(STYLE_VARIANT_OPTIONS)

    # Etichetare in functie de pozitia congelatorului.
    label = 0 if freezer_position == 0 else 1

    return {
        "fridge_height": fridge_height,
        "fridge_width": fridge_width,
        "fridge_depth": fridge_depth,
        "door_thickness": door_thickness,
        "handle_length": handle_length,
        "freezer_ratio": freezer_ratio,
        "freezer_position": freezer_position,
        "style_variant": style_variant,
        "label": label,
    }


def main() -> None:
    """Genereaza dataset-ul si il salveaza in CSV."""
    # Genereaza dataset-ul sintetic si il salveaza in CSV.
    samples = [generate_sample() for _ in range(NUM_SAMPLES)]
    df = pd.DataFrame(samples)

    os.makedirs("data/generated", exist_ok=True)
    df.to_csv("data/generated/fridges_dataset.csv", index=False)

    print("Fridge dataset generated.")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
