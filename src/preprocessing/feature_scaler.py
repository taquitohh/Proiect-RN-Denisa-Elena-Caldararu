"""Feature scaling for the cleaned chair dataset."""

import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


INPUT_PATH = os.path.join("data", "processed", "chairs_clean.csv")
OUTPUT_PATH = os.path.join("data", "processed", "chairs_scaled.csv")
SCALER_PATH = os.path.join("config", "preprocessing_params.pkl")


def main() -> None:
    """Scale features with StandardScaler and save outputs."""
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    feature_columns = [
        "seat_height",
        "seat_width",
        "seat_depth",
        "leg_count",
        "leg_thickness",
        "has_backrest",
        "backrest_height",
        "style_variant",
    ]

    X = df[feature_columns]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    scaled_df["label"] = y.values

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    scaled_df.to_csv(OUTPUT_PATH, index=False)
    with open(SCALER_PATH, "wb") as file_handle:
        pickle.dump(scaler, file_handle)

    print("Scaling complete.")
    print(f"Saved scaled dataset to {OUTPUT_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
