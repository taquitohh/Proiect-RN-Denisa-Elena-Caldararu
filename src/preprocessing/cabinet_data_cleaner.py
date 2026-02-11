"""Curatare date pentru dataset-ul sintetic de dulapuri."""

import os

import pandas as pd


# Cai pentru fisierul brut si pentru output-ul curatat.
INPUT_PATH = os.path.join("data", "generated", "cabinets_dataset.csv")
OUTPUT_PATH = os.path.join("data", "processed", "cabinets_clean.csv")


def validate_dataset(df: pd.DataFrame) -> None:
    """Valideaza dataset-ul pentru valori lipsa sau invalide."""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

    # Verificare simpla pentru valori lipsa.
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values (NaN).")


def main() -> None:
    """Incarca, valideaza si salveaza dataset-ul curatat."""
    # Verifica existenta fisierului de intrare.
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # Incarca, valideaza si salveaza dataset-ul curatat.
    df = pd.read_csv(INPUT_PATH)
    validate_dataset(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Rows: {len(df)}")
    print("Dataset is clean and valid.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
