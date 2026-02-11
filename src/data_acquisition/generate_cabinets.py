"""Genereaza dataset sintetic pentru clasificarea dulapurilor (Etapa 4).

Etichete:
0 = single_door, 1 = double_door, 2 = tall_cabinet
"""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.


from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

# Dimensiune dataset si seed pentru reproductibilitate.
NUM_SAMPLES = 12000
RANDOM_SEED = 42

CABINET_HEIGHT_RANGE = (1.0, 2.2)
CABINET_WIDTH_RANGE = (0.6, 1.6)
CABINET_DEPTH_RANGE = (0.3, 0.8)
WALL_THICKNESS_RANGE = (0.015, 0.05)
DOOR_TYPE_OPTIONS = (0, 1)  # 0 = flush, 1 = inset
DOOR_COUNT_OPTIONS = (1, 2)
STYLE_VARIANT_OPTIONS = (0, 1, 2)

# Initializeaza RNG-urile o singura data.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_sample() -> dict[str, float | int]:
    """Genereaza un sample de dulap cu valori valide."""
    # Genereaza valori in intervalele definite.
    cabinet_height = random.uniform(*CABINET_HEIGHT_RANGE)
    cabinet_width = random.uniform(*CABINET_WIDTH_RANGE)
    cabinet_depth = random.uniform(*CABINET_DEPTH_RANGE)
    wall_thickness = random.uniform(*WALL_THICKNESS_RANGE)
    door_type = random.choice(DOOR_TYPE_OPTIONS)
    door_count = random.choice(DOOR_COUNT_OPTIONS)
    style_variant = random.choice(STYLE_VARIANT_OPTIONS)

    # Etichetare pe baza inaltimii si a numarului de usi.
    if cabinet_height > 1.8:
        label = 2
    elif door_count == 1:
        label = 0
    else:
        label = 1

    return {
        "cabinet_height": cabinet_height,
        "cabinet_width": cabinet_width,
        "cabinet_depth": cabinet_depth,
        "wall_thickness": wall_thickness,
        "door_type": door_type,
        "door_count": door_count,
        "style_variant": style_variant,
        "label": label,
    }


def main() -> None:
    """Genereaza dataset-ul si il salveaza in CSV."""
    # Genereaza dataset-ul sintetic si il salveaza in CSV.
    samples = [generate_sample() for _ in range(NUM_SAMPLES)]
    df = pd.DataFrame(samples)

    os.makedirs("data/generated", exist_ok=True)
    df.to_csv("data/generated/cabinets_dataset.csv", index=False)

    print("Cabinet dataset generated.")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
