"""Split the scaled chair dataset into train/validation/test sets."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = os.path.join("data", "processed", "chairs_scaled.csv")
TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "validation")
TEST_DIR = os.path.join("data", "test")

RANDOM_STATE = 42


def save_split(features: pd.DataFrame, labels: pd.Series, output_dir: str, prefix: str) -> None:
    """Save features and labels to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    features.to_csv(os.path.join(output_dir, f"X_{prefix}.csv"), index=False)
    labels.to_csv(os.path.join(output_dir, f"y_{prefix}.csv"), index=False)


def main() -> None:
    """Load scaled data, split it, and save CSV files."""
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

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
