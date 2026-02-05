"""Flask UI for chair type classification and Blender script generation."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
from flask import Flask, render_template_string, request
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blender_scripts.chair_generator import generate_chair_script


LABEL_MAP = {
    0: "Simple Chair",
    1: "Chair with Backrest",
    2: "Bar Chair",
    3: "Stool",
}

SCALER_PATH = os.path.join("config", "preprocessing_params.pkl")
MODEL_PATH = os.path.join("models", "trained_model.h5")


def load_scaler(path: str):
    """Load the preprocessing scaler from disk."""
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def load_model(path: str) -> keras.Model:
    """Load the trained Keras model from disk."""
    return keras.models.load_model(path)


def load_artifacts() -> tuple[keras.Model, object]:
    """Load model and scaler once at app startup."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Please run preprocessing first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train the model first.")

    print(f"Loading model: {MODEL_PATH}")
    scaler = load_scaler(SCALER_PATH)
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model, scaler


def build_input_array(values) -> np.ndarray:
    """Build a 2D numpy array from user inputs."""
    return np.array([values], dtype=float)


app = Flask(__name__)

MODEL, SCALER = load_artifacts()


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Chair Type Classifier (Etapa 5.4)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
            label { font-weight: 600; display: block; margin-bottom: 4px; }
            input, select { width: 100%; padding: 6px; }
            .box { margin-top: 16px; padding: 12px; border: 1px solid #ddd; }
            pre { background: #f6f6f6; padding: 12px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>Chair Type Classifier (Etapa 5.4)</h1>
        <p>Introduceți parametrii geometrici ai scaunului și apăsați Predict.</p>

        <form method="post">
            <div class="grid">
                <div>
                    <label for="seat_height">seat_height</label>
                    <input type="number" step="0.01" min="0.4" max="0.8" name="seat_height" id="seat_height" value="{{ values.seat_height }}" required />
                </div>
                <div>
                    <label for="seat_width">seat_width</label>
                    <input type="number" step="0.01" min="0.35" max="0.6" name="seat_width" id="seat_width" value="{{ values.seat_width }}" required />
                </div>
                <div>
                    <label for="seat_depth">seat_depth</label>
                    <input type="number" step="0.01" min="0.35" max="0.6" name="seat_depth" id="seat_depth" value="{{ values.seat_depth }}" required />
                </div>
                <div>
                    <label for="leg_count">leg_count</label>
                    <input type="number" step="1" min="3" max="5" name="leg_count" id="leg_count" value="{{ values.leg_count }}" required />
                </div>
                <div>
                    <label for="leg_thickness">leg_thickness</label>
                    <input type="number" step="0.01" min="0.03" max="0.08" name="leg_thickness" id="leg_thickness" value="{{ values.leg_thickness }}" required />
                </div>
                <div>
                    <label for="has_backrest">has_backrest</label>
                    <select name="has_backrest" id="has_backrest">
                        <option value="0" {% if values.has_backrest == 0 %}selected{% endif %}>0</option>
                        <option value="1" {% if values.has_backrest == 1 %}selected{% endif %}>1</option>
                    </select>
                </div>
                <div>
                    <label for="backrest_height">backrest_height</label>
                    <input type="number" step="0.01" min="0.2" max="0.5" name="backrest_height" id="backrest_height" value="{{ values.backrest_height }}" required />
                </div>
                <div>
                    <label for="style_variant">style_variant</label>
                    <input type="number" step="1" min="0" max="2" name="style_variant" id="style_variant" value="{{ values.style_variant }}" required />
                </div>
            </div>

            <div style="margin-top: 12px;">
                <button type="submit">Predict</button>
            </div>
        </form>

        {% if result %}
            <div class="box">
                <div><strong>Predicted class:</strong> {{ result.label }}</div>
                <div><strong>Confidence:</strong> {{ "%.2f" | format(result.confidence) }}</div>
                <div><strong>Probabilities:</strong></div>
                <pre>{{ result.probabilities | tojson(indent=2) }}</pre>
                <div><strong>Generated Blender script:</strong></div>
                <pre>{{ result.script }}</pre>
            </div>
        {% endif %}

        <script>
            const hasBackrest = document.getElementById('has_backrest');
            const backrestHeight = document.getElementById('backrest_height');

            function syncBackrest() {
                if (hasBackrest.value === '0') {
                    backrestHeight.value = '0.0';
                    backrestHeight.min = '0.0';
                    backrestHeight.max = '0.0';
                    backrestHeight.setAttribute('readonly', 'readonly');
                } else {
                    backrestHeight.min = '0.2';
                    backrestHeight.max = '0.5';
                    if (parseFloat(backrestHeight.value) < 0.2) {
                        backrestHeight.value = '0.2';
                    }
                    backrestHeight.removeAttribute('readonly');
                }
            }

            hasBackrest.addEventListener('change', syncBackrest);
            syncBackrest();
        </script>
    </body>
</html>
"""


def parse_float(name: str, default: float) -> float:
        value = request.form.get(name, default)
        return float(value)


def parse_int(name: str, default: int) -> int:
        value = request.form.get(name, default)
        return int(value)


@app.route("/", methods=["GET", "POST"])
def index():
        values = {
                "seat_height": 0.55,
                "seat_width": 0.45,
                "seat_depth": 0.45,
                "leg_count": 4,
                "leg_thickness": 0.05,
                "has_backrest": 1,
                "backrest_height": 0.25,
                "style_variant": 0,
        }

        result = None
        if request.method == "POST":
                values["seat_height"] = parse_float("seat_height", values["seat_height"])
                values["seat_width"] = parse_float("seat_width", values["seat_width"])
                values["seat_depth"] = parse_float("seat_depth", values["seat_depth"])
                values["leg_count"] = parse_int("leg_count", values["leg_count"])
                values["leg_thickness"] = parse_float("leg_thickness", values["leg_thickness"])
                values["has_backrest"] = parse_int("has_backrest", values["has_backrest"])
                values["backrest_height"] = parse_float("backrest_height", values["backrest_height"])
                values["style_variant"] = parse_int("style_variant", values["style_variant"])

                if values["has_backrest"] == 0:
                        values["backrest_height"] = 0.0
                else:
                        values["backrest_height"] = max(0.2, min(0.5, values["backrest_height"]))

                features = build_input_array(
                        [
                                values["seat_height"],
                                values["seat_width"],
                                values["seat_depth"],
                                values["leg_count"],
                                values["leg_thickness"],
                                values["has_backrest"],
                                values["backrest_height"],
                                values["style_variant"],
                        ]
                )
                scaled_features = SCALER.transform(features)
                probabilities = MODEL.predict(scaled_features, verbose=0)[0]
                predicted_label = int(np.argmax(probabilities))
                confidence = float(np.max(probabilities))

                script = generate_chair_script(
                        seat_height=values["seat_height"],
                        seat_width=values["seat_width"],
                        seat_depth=values["seat_depth"],
                        leg_count=values["leg_count"],
                        leg_thickness=values["leg_thickness"],
                        has_backrest=values["has_backrest"],
                        backrest_height=values["backrest_height"],
                        style_variant=values["style_variant"],
                )

                result = {
                        "label": LABEL_MAP[predicted_label],
                        "confidence": confidence,
                        "probabilities": {LABEL_MAP[idx]: float(prob) for idx, prob in enumerate(probabilities)},
                        "script": script,
                }

        return render_template_string(HTML_TEMPLATE, values=values, result=result)


if __name__ == "__main__":
        app.run(host="127.0.0.1", port=5000, debug=True)
