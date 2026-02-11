"""Imparte dataset-ul scalat in train/validation/test."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split


# Cai pentru input scalat si pentru folderele de split.
INPUT_PATH = os.path.join("data", "processed", "tables_scaled.csv")
TRAIN_DIR = os.path.join("data", "tables", "train")
VAL_DIR = os.path.join("data", "tables", "validation")
TEST_DIR = os.path.join("data", "tables", "test")

RANDOM_STATE = 42


def save_split(features: pd.DataFrame, labels: pd.Series, output_dir: str, prefix: str) -> None:
    """Salveaza features si label-uri in CSV."""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

    # Salveaza spliturile in format consistent pentru pipeline.
    os.makedirs(output_dir, exist_ok=True)
    features.to_csv(os.path.join(output_dir, f"X_{prefix}.csv"), index=False)
    labels.to_csv(os.path.join(output_dir, f"y_{prefix}.csv"), index=False)


def main() -> None:
    """Incarca datele scalate, le imparte si salveaza CSV-urile."""
    # Verifica existenta dataset-ului scalat.
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    # Primul split: train vs temp (val+test), cu stratificare.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Al doilea split: validation vs test, cu stratificare.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    save_split(X_train, y_train, TRAIN_DIR, "train")
    save_split(X_val, y_val, VAL_DIR, "val")
    save_split(X_test, y_test, TEST_DIR, "test")

    print("Split complete.")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    print("Stratified split used.")


if __name__ == "__main__":
    main()
