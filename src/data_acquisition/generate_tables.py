"""Genereaza dataset sintetic pentru clasificarea meselor (Etapa 4).

Etichete:
0 = low_table, 1 = dining_table, 2 = bar_table
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

TABLE_HEIGHT_RANGE = (0.35, 0.9)
TABLE_WIDTH_RANGE = (0.5, 1.4)
TABLE_DEPTH_RANGE = (0.5, 1.0)
LEG_COUNT_OPTIONS = (3, 4)
LEG_THICKNESS_RANGE = (0.04, 0.09)
HAS_APRON_OPTIONS = (0, 1)
STYLE_VARIANT_OPTIONS = (0, 1, 2)

# Initializeaza RNG-urile o singura data.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_sample() -> dict[str, float | int]:
    """Genereaza un sample de masa cu valori valide."""
    # Genereaza valori in intervalele definite.
    table_height = random.uniform(*TABLE_HEIGHT_RANGE)
    table_width = random.uniform(*TABLE_WIDTH_RANGE)
    table_depth = random.uniform(*TABLE_DEPTH_RANGE)
    leg_count = random.choice(LEG_COUNT_OPTIONS)
    leg_thickness = random.uniform(*LEG_THICKNESS_RANGE)
    has_apron = random.choice(HAS_APRON_OPTIONS)
    style_variant = random.choice(STYLE_VARIANT_OPTIONS)

    # Etichetare pe baza pragurilor de inaltime.
    if table_height < 0.45:
        label = 0
    elif table_height <= 0.75:
        label = 1
    else:
        label = 2

    return {
        "table_height": table_height,
        "table_width": table_width,
        "table_depth": table_depth,
        "leg_count": leg_count,
        "leg_thickness": leg_thickness,
        "has_apron": has_apron,
        "style_variant": style_variant,
        "label": label,
    }


def main() -> None:
    """Genereaza dataset-ul si il salveaza in CSV."""
    # Genereaza dataset-ul sintetic si il salveaza in CSV.
    samples = [generate_sample() for _ in range(NUM_SAMPLES)]
    df = pd.DataFrame(samples)

    os.makedirs("data/generated", exist_ok=True)
    df.to_csv("data/generated/tables_dataset.csv", index=False)

    print("Table dataset generated.")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
