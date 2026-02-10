"""Scalare features pentru dataset-ul curatat de plite."""

import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Cai pentru dataset-ul scalat si pentru scaler-ul salvat.
INPUT_PATH = os.path.join("data", "processed", "stoves_clean.csv")
OUTPUT_PATH = os.path.join("data", "processed", "stoves_scaled.csv")
SCALER_PATH = os.path.join("config", "stove_scaler.pkl")


def main() -> None:
    """Scaleaza features cu StandardScaler si salveaza output-urile."""
    # Verifica existenta dataset-ului curatat inainte de scalare.
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Coloanele numerice folosite de model la antrenare/inferenta.
    feature_columns = [
        "stove_height",
        "stove_width",
        "stove_depth",
        "oven_height_ratio",
        "handle_length",
        "glass_thickness",
        "style_variant",
        "burner_count",
        "knob_count",
    ]

    X = df[feature_columns]
    y = df["label"]

    # Antreneaza scaler-ul pe features si aplica transformarea.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Salveaza dataset-ul scalat si scaler-ul pentru inferenta.
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
