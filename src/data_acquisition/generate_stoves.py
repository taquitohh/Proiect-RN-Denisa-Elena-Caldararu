"""Genereaza dataset sintetic pentru clasificarea plitelor."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

# Dimensiune dataset si seed pentru reproductibilitate.
NUM_SAMPLES = 12000
RANDOM_SEED = 42

STOVE_HEIGHT_RANGE = (0.85, 0.95)
STOVE_WIDTH_RANGE = (0.6, 0.9)
STOVE_DEPTH_RANGE = (0.6, 0.75)
OVEN_HEIGHT_RATIO_RANGE = (0.45, 0.6)
HANDLE_LENGTH_RANGE = (0.35, 0.65)
GLASS_THICKNESS_RANGE = (0.01, 0.03)
STYLE_VARIANT_OPTIONS = (0, 1, 2)

# Initializeaza RNG-urile o singura data.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_sample() -> dict[str, float | int]:
    """Genereaza un sample de plita cu valori valide."""
    # Genereaza valori in intervalele definite.
    stove_height = random.uniform(*STOVE_HEIGHT_RANGE)
    stove_width = random.uniform(*STOVE_WIDTH_RANGE)
    stove_depth = random.uniform(*STOVE_DEPTH_RANGE)
    oven_height_ratio = random.uniform(*OVEN_HEIGHT_RATIO_RANGE)
    handle_length = random.uniform(*HANDLE_LENGTH_RANGE)
    glass_thickness = random.uniform(*GLASS_THICKNESS_RANGE)
    style_variant = random.choice(STYLE_VARIANT_OPTIONS)

    # Etichetare pe baza pragurilor de grosime a sticlei.
    if glass_thickness < 0.017:
        label = 0
    elif glass_thickness <= 0.024:
        label = 1
    else:
        label = 2

    return {
        "stove_height": stove_height,
        "stove_width": stove_width,
        "stove_depth": stove_depth,
        "oven_height_ratio": oven_height_ratio,
        "handle_length": handle_length,
        "glass_thickness": glass_thickness,
        "style_variant": style_variant,
        "burner_count": 4,
        "knob_count": 6,
        "label": label,
    }


def main() -> None:
    """Genereaza dataset-ul si il salveaza in CSV."""
    # Genereaza dataset-ul sintetic si il salveaza in CSV.
    samples = [generate_sample() for _ in range(NUM_SAMPLES)]
    df = pd.DataFrame(samples)

    os.makedirs("data/generated", exist_ok=True)
    df.to_csv("data/generated/stoves_dataset.csv", index=False)

    print("Stove dataset generated.")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
