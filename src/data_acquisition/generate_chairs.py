"""Generate a synthetic chair dataset for neural network training.

This script creates a reproducible dataset of chair parameters and labels
based on deterministic rules defined in the project specification.
"""

import os
import random
from typing import Dict

import numpy as np
import pandas as pd

# Global parameters
NUM_SAMPLES = 15000
RANDOM_SEED = 42

# Feature ranges and options
SEAT_HEIGHT_RANGE = (0.4, 0.8)
SEAT_WIDTH_RANGE = (0.35, 0.6)
SEAT_DEPTH_RANGE = (0.35, 0.6)
LEG_COUNT_OPTIONS = (3, 4, 5)
LEG_THICKNESS_RANGE = (0.03, 0.08)
HAS_BACKREST_OPTIONS = (0, 1)
BACKREST_HEIGHT_RANGE = (0.2, 0.5)
STYLE_VARIANT_OPTIONS = (0, 1, 2)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_sample() -> Dict[str, float]:
    """Generate a single chair sample with valid feature values."""
    seat_height = random.uniform(*SEAT_HEIGHT_RANGE)
    seat_width = random.uniform(*SEAT_WIDTH_RANGE)
    seat_depth = random.uniform(*SEAT_DEPTH_RANGE)
    leg_count = random.choice(LEG_COUNT_OPTIONS)
    leg_thickness = random.uniform(*LEG_THICKNESS_RANGE)
    has_backrest = random.choice(HAS_BACKREST_OPTIONS)

    if has_backrest == 1:
        backrest_height = random.uniform(*BACKREST_HEIGHT_RANGE)
    else:
        backrest_height = 0.0

    style_variant = random.choice(STYLE_VARIANT_OPTIONS)

    return {
        "seat_height": seat_height,
        "seat_width": seat_width,
        "seat_depth": seat_depth,
        "leg_count": leg_count,
        "leg_thickness": leg_thickness,
        "has_backrest": has_backrest,
        "backrest_height": backrest_height,
        "style_variant": style_variant,
    }


def assign_label(sample: Dict[str, float]) -> int:
    """Assign label based on deterministic rules and required order."""
    seat_height = sample["seat_height"]
    has_backrest = sample["has_backrest"]
    backrest_height = sample["backrest_height"]

    if seat_height > 0.65:
        return 2  # bar_chair
    if has_backrest == 0 and seat_height < 0.5:
        return 3  # stool
    if has_backrest == 1 and backrest_height >= 0.25 and seat_height <= 0.65:
        return 1  # chair_with_backrest
    return 0  # chair_simple


def build_dataset(num_samples: int) -> pd.DataFrame:
    """Build the full dataset as a pandas DataFrame."""
    samples = []
    for _ in range(num_samples):
        sample = generate_sample()
        sample["label"] = assign_label(sample)
        samples.append(sample)

    df = pd.DataFrame(samples)

    # Enforce column order
    column_order = [
        "seat_height",
        "seat_width",
        "seat_depth",
        "leg_count",
        "leg_thickness",
        "has_backrest",
        "backrest_height",
        "style_variant",
        "label",
    ]
    return df[column_order]


def validate_dataset(df: pd.DataFrame) -> None:
    """Run minimal validations required by specification."""
    invalid_backrest = df[(df["has_backrest"] == 0) & (df["backrest_height"] != 0)]
    if not invalid_backrest.empty:
        raise ValueError("Invalid rows found: backrest_height must be 0 when has_backrest is 0.")


def ensure_output_path(output_path: str) -> None:
    """Create output directories if they do not exist."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def main() -> None:
    """Generate dataset and save it to CSV."""
    set_random_seed(RANDOM_SEED)

    dataset = build_dataset(NUM_SAMPLES)
    validate_dataset(dataset)

    output_path = os.path.join("data", "generated", "chairs_dataset.csv")
    ensure_output_path(output_path)

    dataset.to_csv(output_path, index=False)

    print("Dataset generated successfully.")
    print(f"Total samples: {len(dataset)}")
    print("Class distribution:")
    class_counts = dataset["label"].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"{label}: {count}")
    print(f"Saved to {output_path}")
    print("Preview:")
    print(dataset.head())


if __name__ == "__main__":
    main()
