"""Flask UI for chair type classification and Blender script generation."""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
import urllib.error
import urllib.request
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
BLENDER_API_URL = os.environ.get("BLENDER_API_URL", "http://127.0.0.1:5001/render")


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
            :root {
                --bg: #0e1116;
                --ink: #f2f4f8;
                --muted: #9aa3af;
                --paper: #151a22;
                --accent: #4f8c7a;
                --accent-2: #c97a5a;
                --shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
                --border: rgba(255, 255, 255, 0.08);
            }

            * { box-sizing: border-box; }

            body {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                margin: 0;
                background: radial-gradient(1200px 400px at 20% -10%, rgba(79, 140, 122, 0.35) 0%, rgba(79, 140, 122, 0) 60%),
                    radial-gradient(900px 500px at 100% 0%, rgba(201, 122, 90, 0.35) 0%, rgba(201, 122, 90, 0) 55%),
                    var(--bg);
                color: var(--ink);
            }

            .page {
                max-width: 1100px;
                margin: 40px auto 64px;
                padding: 0 20px;
                animation: fadeIn 600ms ease-out;
            }

            .hero {
                display: grid;
                grid-template-columns: minmax(260px, 1.2fr) minmax(220px, 0.8fr);
                gap: 24px;
                align-items: end;
                margin-bottom: 24px;
            }

            .title {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: clamp(28px, 4vw, 40px);
                margin: 0 0 8px;
                letter-spacing: -0.02em;
            }

            .subtitle {
                margin: 0;
                max-width: 540px;
                color: var(--muted);
            }

            .badge {
                justify-self: end;
                padding: 10px 16px;
                border-radius: 999px;
                background: var(--accent);
                color: #fff;
                font-weight: 600;
                box-shadow: var(--shadow);
            }

            .card {
                background: var(--paper);
                border-radius: 20px;
                padding: 20px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border);
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 14px;
            }

            label {
                font-weight: 600;
                display: block;
                margin-bottom: 6px;
            }

            input, select {
                width: 100%;
                padding: 9px 10px;
                border-radius: 10px;
                border: 1px solid var(--border);
                background: #0f141b;
                color: var(--ink);
                font-size: 14px;
            }

            button {
                border: 0;
                background: var(--accent);
                color: #fff;
                padding: 10px 16px;
                border-radius: 12px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: var(--shadow);
                transition: transform 120ms ease, background 120ms ease;
            }

            button:hover { transform: translateY(-1px); background: #1f5d46; }

            .box {
                margin-top: 18px;
                padding: 16px;
                border-radius: 16px;
                background: #0f141b;
                border: 1px solid var(--border);
            }

            .result-title {
                font-size: clamp(20px, 3vw, 28px);
                margin: 0 0 8px;
                color: var(--ink);
            }

            .output-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 10px;
            }

            .script-wrap {
                position: relative;
            }

            .copy-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(15, 20, 27, 0.9);
                border: 1px solid var(--border);
                color: var(--ink);
                padding: 6px 8px;
                border-radius: 10px;
                box-shadow: var(--shadow);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 34px;
                height: 30px;
            }

            .copy-btn svg {
                width: 16px;
                height: 16px;
                fill: var(--ink);
            }

            pre {
                background: #0b0f14;
                color: #e6e9ef;
                padding: 14px;
                border-radius: 12px;
                overflow-x: auto;
                font-size: 13px;
                line-height: 1.5;
            }

            .preview {
                margin-top: 14px;
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid var(--border);
                background: #0b0f14;
            }

            .preview img {
                width: 100%;
                display: block;
            }

            .preview-note {
                color: var(--muted);
                font-size: 12px;
                margin-top: 8px;
            }

            .section-title {
                font-weight: 700;
                margin: 10px 0 8px;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 11px;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 720px) {
                .hero { grid-template-columns: 1fr; }
                .badge { justify-self: start; }
            }
        </style>
    </head>
    <body>
        <div class="page">
            <div class="hero">
                <div>
                    <h1 class="title">Chair Type Classifier (Etapa 5.4)</h1>
                    <p class="subtitle">Introduceți parametrii geometrici ai scaunului și apăsați Predict.</p>
                </div>
                <div class="badge">RN + Blender Script</div>
            </div>

            <form method="post" class="card">
                <div class="grid">
                    <div>
                        <label for="object_type">object_type</label>
                        <select name="object_type" id="object_type" onchange="syncObjectType()">
                            <option value="chair" selected>chair</option>
                            <option value="table" disabled>table (coming soon)</option>
                            <option value="cabinet" disabled>cabinet (coming soon)</option>
                        </select>
                    </div>
                </div>

                <div class="section-title">Sezut</div>
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
                </div>

                <div class="section-title">Picioare</div>
                <div class="grid">
                    <div>
                        <label for="leg_count">leg_count</label>
                        <input type="number" step="1" min="3" max="5" name="leg_count" id="leg_count" value="{{ values.leg_count }}" required />
                    </div>
                    <div>
                        <label for="leg_shape">leg_shape</label>
                        <select name="leg_shape" id="leg_shape">
                            <option value="square" {% if values.leg_shape == "square" %}selected{% endif %}>square</option>
                            <option value="round" {% if values.leg_shape == "round" %}selected{% endif %}>round</option>
                        </select>
                    </div>
                    <div>
                        <label for="leg_size">leg_size</label>
                        <input type="number" step="0.01" min="0.03" max="0.08" name="leg_size" id="leg_size" value="{{ values.leg_size }}" required />
                    </div>
                </div>

                <div class="section-title">Spatar</div>
                <div class="grid">
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
                    <div class="result-title">Predicted variant: {{ result.label }}</div>
                    <div><strong>Confidence:</strong> {{ "%.2f" | format(result.confidence) }}</div>
                    <div><strong>Probabilities:</strong></div>
                    <pre>{{ result.probabilities | tojson(indent=2) }}</pre>
                    <div class="output-header">
                        <div><strong>Generated Blender script:</strong></div>
                    </div>
                    <div class="script-wrap">
                        <button class="copy-btn" type="button" onclick="copyScript()" aria-label="Copy script">
                            <svg viewBox="0 0 24 24" role="img" aria-hidden="true">
                                <path d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H10V7h9v14z"/>
                            </svg>
                        </button>
                        <pre id="blender-script">{{ result.script }}</pre>
                    </div>

                    {% if result.preview_src %}
                        <div class="preview">
                            <img src="{{ result.preview_src }}" alt="Blender preview" />
                        </div>
                    {% elif result.preview_error %}
                        <div class="preview-note">Preview error: {{ result.preview_error }}</div>
                    {% else %}
                        <div class="preview-note">Preview: pornește Blender API pe 127.0.0.1:5001.</div>
                    {% endif %}
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

            function syncObjectType() {
                const objectType = document.getElementById('object_type');
                if (!objectType || objectType.value !== 'chair') {
                    alert('Doar scaunul este implementat momentan.');
                    objectType.value = 'chair';
                }
            }

            function copyScript() {
                const script = document.getElementById('blender-script');
                if (!script) {
                    return;
                }
                navigator.clipboard.writeText(script.textContent || '')
                    .then(() => alert('Script copied to clipboard.'))
                    .catch(() => alert('Copy failed.'));
            }
        </script>
        </div>
    </body>
</html>
"""


def parse_float(name: str, default: float) -> float:
        value = request.form.get(name, default)
        return float(value)


def parse_int(name: str, default: int) -> int:
        value = request.form.get(name, default)
        return int(value)


def parse_str(name: str, default: str) -> str:
    value = request.form.get(name, default)
    return str(value)


def request_preview(payload: dict) -> tuple[str | None, str | None]:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            BLENDER_API_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body.get("image_base64"), None
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, str(exc)


@app.route("/", methods=["GET", "POST"])
def index():
        values = {
                "seat_height": 0.55,
                "seat_width": 0.45,
                "seat_depth": 0.45,
                "leg_count": 4,
            "leg_shape": "square",
            "leg_size": 0.05,
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
                values["leg_shape"] = parse_str("leg_shape", values["leg_shape"])
                values["leg_size"] = parse_float("leg_size", values["leg_size"])
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
                                values["leg_size"],
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
                        leg_shape=values["leg_shape"],
                        leg_size=values["leg_size"],
                        has_backrest=values["has_backrest"],
                        backrest_height=values["backrest_height"],
                        style_variant=values["style_variant"],
                )

                preview_payload = {
                    "seat_height": values["seat_height"],
                    "seat_width": values["seat_width"],
                    "seat_depth": values["seat_depth"],
                    "leg_count": values["leg_count"],
                    "leg_shape": values["leg_shape"],
                    "leg_size": values["leg_size"],
                    "has_backrest": values["has_backrest"],
                    "backrest_height": values["backrest_height"],
                    "style_variant": values["style_variant"],
                }
                preview_image, preview_error = request_preview(preview_payload)
                preview_src = None
                if preview_image:
                    preview_src = f"data:image/png;base64,{preview_image}"

                result = {
                        "label": LABEL_MAP[predicted_label],
                        "confidence": confidence,
                        "probabilities": {LABEL_MAP[idx]: float(prob) for idx, prob in enumerate(probabilities)},
                    "script": script,
                    "preview_src": preview_src,
                    "preview_error": preview_error,
                }

        return render_template_string(HTML_TEMPLATE, values=values, result=result)


if __name__ == "__main__":
        app.run(host="127.0.0.1", port=5000, debug=True)
