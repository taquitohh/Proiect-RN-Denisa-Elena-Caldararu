"""Genereaza un GIF demo end-to-end folosind un sample real din test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "docs" / "demo" / "demo_end_to_end.gif"
MODEL_PATH = PROJECT_ROOT / "models" / "chair_model.h5"
TEST_X_PATH = PROJECT_ROOT / "data" / "chairs" / "test" / "X_test.csv"
TEST_Y_PATH = PROJECT_ROOT / "data" / "chairs" / "test" / "y_test.csv"

LABEL_NAMES = {
    0: "chair_simple",
    1: "chair_with_backrest",
    2: "bar_chair",
    3: "stool",
}


def load_sample() -> tuple[pd.Series, int, int, float]:
    """Incarca un sample de test si returneaza (features, true, pred, confidence)."""
    x_test = pd.read_csv(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH).squeeze().to_numpy()

    model = tf.keras.models.load_model(MODEL_PATH)
    sample = x_test.iloc[0]
    probs = model.predict(sample.to_frame().T, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    true = int(y_test[0])
    return sample, true, pred, conf


def render_frame(title: str, lines: list[str]) -> Image.Image:
    """Randeaza un frame cu continut text."""
    width, height = 640, 360
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_title = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        font_title = font

    draw.text((20, 15), title, fill="black", font=font_title)

    y = 60
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += 22

    return img


def main() -> None:
    sample, true, pred, conf = load_sample()

    feature_lines = [f"{name}: {value:.3f}" for name, value in sample.items()]
    feature_lines = feature_lines[:10]

    frames = [
        render_frame(
            "Pas 1 - Input features",
            ["Sample din data/chairs/test:"] + feature_lines,
        ),
        render_frame(
            "Pas 2 - Preprocesare",
            [
                "StandardScaler aplicat (config/chair_scaler.pkl)",
                "Features aliniate la schema de antrenare.",
            ],
        ),
        render_frame(
            "Pas 3 - Inferenta",
            [
                f"Eticheta reala: {LABEL_NAMES.get(true, str(true))}",
                f"Prezisa: {LABEL_NAMES.get(pred, str(pred))}",
                f"Incredere: {conf:.3f}",
            ],
        ),
        render_frame(
            "Pas 4 - Output",
            [
                "Rezultatul este afisat in UI.",
                "Script Blender generat pentru clasa prezisa.",
            ],
        ),
    ]

    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        DEMO_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
    )


if __name__ == "__main__":
    main()
