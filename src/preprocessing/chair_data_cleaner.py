"""Data cleaning for the synthetic chair dataset."""

import os

import pandas as pd


INPUT_PATH = os.path.join("data", "generated", "chairs_dataset.csv")
OUTPUT_PATH = os.path.join("data", "processed", "chairs_clean.csv")


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset for missing or invalid values."""
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values (NaN).")

    invalid_backrest = df[(df["has_backrest"] == 0) & (df["backrest_height"] > 0)]
    if not invalid_backrest.empty:
        raise ValueError("Invalid rows: backrest_height > 0 when has_backrest == 0.")


def main() -> None:
    """Load, validate, and save the cleaned dataset."""
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    validate_dataset(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Rows: {len(df)}")
    print("Dataset is clean and valid.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
