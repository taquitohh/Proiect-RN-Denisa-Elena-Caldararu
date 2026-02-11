"""UI Flask pentru clasificare si generare de script Blender."""

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
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from tensorflow import keras

# Permite rularea aplicatiei din orice director prin setarea sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blender_scripts.chair_generator import generate_chair_script
from src.blender_scripts.table_generator import generate_table_script
from src.blender_scripts.cabinet_generator import generate_cabinet_script
from src.blender_scripts.fridge_generator import generate_fridge_script
from src.blender_scripts.stove_generator import generate_stove_script


# Etichete lizibile pentru fiecare tip de obiect.
LABEL_MAPS = {
    "chair": {
        0: "Simple Chair",
        1: "Chair with Backrest",
        2: "Bar Chair",
        3: "Stool",
    },
    "table": {
        0: "Low Table",
        1: "Dining Table",
        2: "Bar Table",
    },
    "cabinet": {
        0: "Single Door",
        1: "Double Door",
        2: "Tall Cabinet",
    },
    "fridge": {
        0: "Top Freezer",
        1: "Bottom Freezer",
    },
    "stove": {
        0: "Gas Stove",
        1: "Electric Stove",
        2: "Induction Stove",
    },
}

# Cai catre scaler pentru fiecare tip de obiect.
SCALER_PATHS = {
    "chair": os.path.join("config", "chair_scaler.pkl"),
    "table": os.path.join("config", "table_scaler.pkl"),
    "cabinet": os.path.join("config", "cabinet_scaler.pkl"),
    "fridge": os.path.join("config", "fridge_scaler.pkl"),
    "stove": os.path.join("config", "stove_scaler.pkl"),
}
# Cai catre modele pentru fiecare tip de obiect.
MODEL_PATHS = {
    "chair": os.path.join("models", "chair_model.h5"),
    "table": os.path.join("models", "table_model.h5"),
    "cabinet": os.path.join("models", "cabinet_model.h5"),
    "fridge": os.path.join("models", "fridge_model.h5"),
    "stove": os.path.join("models", "stove_model.h5"),
}
# Endpoint local pentru Blender API (suprascriere prin env var daca e nevoie).
BLENDER_API_URL = os.environ.get("BLENDER_API_URL", "http://127.0.0.1:5001/render")


def load_scaler(path: str):
    """Incarca scaler-ul de preprocesare de pe disc."""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def load_model(path: str) -> keras.Model:
    """Incarca modelul Keras antrenat de pe disc."""
    return keras.models.load_model(path)


# Cache pentru modele/scalere ca sa evitam reload-ul la fiecare request.
ARTIFACT_CACHE: dict[str, tuple[keras.Model, object]] = {}


def load_artifacts(object_type: str) -> tuple[keras.Model, object]:
    """Incarca modelul si scaler-ul la cerere pentru un tip de obiect."""
    # Returneaza din cache daca artefactele sunt deja incarcate.
    if object_type in ARTIFACT_CACHE:
        return ARTIFACT_CACHE[object_type]

    # Rezolva caile si valideaza tipul obiectului.
    scaler_path = SCALER_PATHS.get(object_type)
    model_path = MODEL_PATHS.get(object_type)
    if not scaler_path or not model_path:
        raise ValueError(f"Unknown object_type: {object_type}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Please run preprocessing first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train the model first.")

    # Incarca artefactele o singura data si le tine in cache.
    print(f"Loading model: {model_path}")
    scaler = load_scaler(scaler_path)
    model = load_model(model_path)
    print("Model loaded successfully.")
    ARTIFACT_CACHE[object_type] = (model, scaler)
    return model, scaler


def check_blender_api(timeout: float = 1.2) -> tuple[bool, str | None]:
    # Request OPTIONS rapid pentru a verifica disponibilitatea Blender API.
    request_obj = urllib.request.Request(BLENDER_API_URL, method="OPTIONS")
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            if response.status >= 400:
                return False, f"HTTP {response.status}"
        return True, None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return False, str(exc)


# Schema de features pentru fiecare tip de obiect (folosita la input).
FEATURE_COLUMNS = {
    "chair": [
        "seat_height",
        "seat_width",
        "seat_depth",
        "leg_count",
        "leg_thickness",
        "has_backrest",
        "backrest_height",
        "style_variant",
    ],
    "table": [
        "table_height",
        "table_width",
        "table_depth",
        "leg_count",
        "leg_thickness",
        "has_apron",
        "style_variant",
    ],
    "cabinet": [
        "cabinet_height",
        "cabinet_width",
        "cabinet_depth",
        "wall_thickness",
        "door_type",
        "door_count",
        "style_variant",
    ],
    "fridge": [
        "fridge_height",
        "fridge_width",
        "fridge_depth",
        "door_thickness",
        "handle_length",
        "freezer_ratio",
        "freezer_position",
        "style_variant",
    ],
    "stove": [
        "stove_height",
        "stove_width",
        "stove_depth",
        "oven_height_ratio",
        "handle_length",
        "glass_thickness",
        "style_variant",
        "burner_count",
        "knob_count",
    ],
}


def build_input_array(object_type: str, values: dict) -> pd.DataFrame:
    """Construieste un DataFrame cu numele de feature-uri pentru scaler."""
    # Mapeaza inputurile UI in ordinea asteptata de model.
    if object_type == "table":
        row = {
            "table_height": values["table_height"],
            "table_width": values["table_width"],
            "table_depth": values["table_depth"],
            "leg_count": values["table_leg_count"],
            "leg_thickness": values["table_leg_thickness"],
            "has_apron": values["table_has_apron"],
            "style_variant": values["table_style_variant"],
        }
    elif object_type == "cabinet":
        row = {
            "cabinet_height": values["cabinet_height"],
            "cabinet_width": values["cabinet_width"],
            "cabinet_depth": values["cabinet_depth"],
            "wall_thickness": values["wall_thickness"],
            "door_type": values["door_type"],
            "door_count": values["door_count"],
            "style_variant": values["cabinet_style_variant"],
        }
    elif object_type == "fridge":
        row = {
            "fridge_height": values["fridge_height"],
            "fridge_width": values["fridge_width"],
            "fridge_depth": values["fridge_depth"],
            "door_thickness": values["door_thickness"],
            "handle_length": values["fridge_handle_length"],
            "freezer_ratio": values["freezer_ratio"],
            "freezer_position": values["freezer_position"],
            "style_variant": values["fridge_style_variant"],
        }
    elif object_type == "stove":
        row = {
            "stove_height": values["stove_height"],
            "stove_width": values["stove_width"],
            "stove_depth": values["stove_depth"],
            "oven_height_ratio": values["oven_height_ratio"],
            "handle_length": values["stove_handle_length"],
            "glass_thickness": values["glass_thickness"],
            "style_variant": values["stove_style_variant"],
            "burner_count": values["burner_count"],
            "knob_count": values["knob_count"],
        }
    else:
        row = {
            "seat_height": values["seat_height"],
            "seat_width": values["seat_width"],
            "seat_depth": values["seat_depth"],
            "leg_count": values["leg_count"],
            "leg_thickness": values["leg_size"],
            "has_backrest": values["has_backrest"],
            "backrest_height": values["backrest_height"],
            "style_variant": values["style_variant"],
        }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS[object_type])


# Instanta Flask.
app = Flask(__name__)


@app.route("/status", methods=["GET"])
def status():
    # Ofera info de status pentru panoul UI.
    blender_ok, blender_error = check_blender_api()
    chair_ok = os.path.exists(SCALER_PATHS["chair"]) and os.path.exists(MODEL_PATHS["chair"])
    return jsonify(
        {
            "ui_ok": True,
            "model_ok": chair_ok,
            "blender_api_ok": blender_ok,
            "blender_api_url": BLENDER_API_URL,
            "blender_api_error": blender_error,
        }
    )


# Template HTML/CSS/JS inline servit de Flask.
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>RN Proiect</title>
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
                background: radial-gradient(1200px 520px at 18% -12%, rgba(79, 140, 122, 0.35) 0%, rgba(79, 140, 122, 0) 62%),
                    radial-gradient(980px 680px at 82% 115%, rgba(201, 122, 90, 0.32) 0%, rgba(201, 122, 90, 0) 60%),
                    linear-gradient(180deg, #0e1116 0%, #0b0f14 100%);
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
                position: relative;
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

            .status-board {
                justify-self: end;
                display: grid;
                gap: 8px;
                padding: 14px 16px;
                border-radius: 18px;
                background: rgba(21, 26, 34, 0.95);
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                min-width: 240px;
            }

            .status-title {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.2em;
                color: var(--muted);
            }

            .status-item {
                display: grid;
                grid-template-columns: 10px 1fr;
                gap: 10px;
                align-items: center;
            }

            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #6f7b8b;
                box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.04);
            }

            .status-dot.ok { background: #4fb48c; }
            .status-dot.down { background: #d26c6c; }
            .status-dot.pending { background: #d1a15f; }

            .status-text {
                display: grid;
                gap: 2px;
                font-size: 13px;
            }

            .status-label {
                font-weight: 600;
                color: var(--ink);
            }

            .status-meta {
                font-size: 12px;
                color: var(--muted);
            }

            .status-note {
                font-size: 12px;
                color: var(--muted);
                line-height: 1.4;
            }

            .card {
                background: var(--paper);
                border-radius: 20px;
                padding: 20px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border);
            }

            .layout {
                display: grid;
                grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.1fr);
                gap: 22px;
                align-items: stretch;
            }

            .panel {
                display: flex;
                flex-direction: column;
                gap: 16px;
                height: 100%;
            }

            .input-panel .grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
            }

            .input-panel input,
            .input-panel select {
                padding: 8px 10px;
            }

            .panel-title {
                font-size: 14px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--muted);
                margin: 0 0 6px;
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
                flex: 1;
                display: flex;
            }

            .copy-btn {
                position: absolute;
                top: 12px;
                right: 12px;
                background: rgba(15, 20, 27, 0.9);
                border: 1px solid var(--border);
                color: var(--ink);
                padding: 0;
                border-radius: 10px;
                box-shadow: var(--shadow);
                display: flex;
                align-items: center;
                justify-content: center;
                width: 36px;
                height: 32px;
                line-height: 1;
                font-size: 0;
                z-index: 2;
            }

            .copy-btn svg {
                width: 16px;
                height: 16px;
                fill: var(--ink);
                display: block;
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

            .code-block {
                max-height: 420px;
                overflow: auto;
                white-space: pre;
            }

            .script-wrap .code-block {
                padding-top: 36px;
                flex: 1;
                max-height: none;
                min-height: 260px;
            }

            .output-card {
                display: flex;
                flex-direction: column;
                gap: 16px;
                height: 100%;
            }

            .output-block {
                background: #0f141b;
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 14px;
            }

            .output-block h3 {
                margin: 0 0 10px;
                font-size: 18px;
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

            .scroll-top {
                position: fixed;
                right: 24px;
                bottom: 24px;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                border: 2px solid rgba(79, 140, 122, 0.55);
                background: rgba(22, 92, 72, 0.92);
                color: var(--ink);
                display: grid;
                place-items: center;
                cursor: pointer;
                box-shadow: var(--shadow);
                opacity: 0;
                pointer-events: none;
                transition: opacity 160ms ease, transform 160ms ease;
                transform: translateY(6px);
                z-index: 100;
                font-size: 22px;
                font-weight: 800;
            }

            .scroll-top svg {
                width: 22px;
                height: 22px;
                fill: var(--ink);
                display: block;
            }

            .scroll-top.show {
                opacity: 1;
                pointer-events: auto;
                transform: translateY(0);
            }

            .section-title {
                font-weight: 700;
                margin: 10px 0 8px;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 11px;
            }

            .footer {
                margin-top: 24px;
                text-align: center;
                padding: 22px 20px;
                background: linear-gradient(180deg, rgba(20, 24, 34, 0.98), rgba(16, 19, 28, 0.98));
            }

            .footer-line {
                font-size: 15px;
                color: var(--muted);
            }

            .footer-name {
                color: var(--ink);
                font-weight: 700;
            }

            .footer-meta {
                margin-top: 8px;
                font-size: 13px;
                color: var(--muted);
                letter-spacing: 0.02em;
            }

            .info-btn {
                border: 1px solid var(--border);
                background: rgba(15, 20, 27, 0.9);
                color: var(--ink);
                border-radius: 999px;
                padding: 0 14px;
                height: 38px;
                font-size: 18px;
                cursor: pointer;
                box-shadow: var(--shadow);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                line-height: 1;
                position: absolute;
                right: 0;
                top: 0;
            }

            .hero-head {
                display: flex;
                align-items: center;
                gap: 14px;
                width: 100%;
            }

            .info-btn .info-label {
                font-size: 11px;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .info-overlay {
                position: fixed;
                inset: 0;
                background: rgba(0, 0, 0, 0.65);
                backdrop-filter: blur(3px);
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 999;
            }

            .info-overlay.show {
                display: flex;
            }

            .info-modal {
                background: var(--paper);
                border-radius: 20px;
                width: min(900px, 92vw);
                max-height: 85vh;
                overflow-y: auto;
                padding: 28px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border);
            }

            .info-modal h2 {
                margin-top: 0;
            }

            .info-close {
                position: sticky;
                top: 0;
                float: right;
                background: none;
                border: none;
                color: var(--ink);
                font-size: 22px;
                cursor: pointer;
            }

            .hidden {
                display: none;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 720px) {
                .hero { grid-template-columns: 1fr; }
                .layout { grid-template-columns: 1fr; }
                .input-panel .grid { grid-template-columns: 1fr; }
            }

            @media (max-width: 1100px) {
                .input-panel .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
        </style>
    </head>
    <body>
        <div id="top"></div>
        <div class="page">
            <div class="hero">
                <div>
                    <div class="hero-head">
                        <h1 class="title">Kitchen Furniture Classifier</h1>
                        <button class="info-btn" onclick="openInfo()" aria-label="Info">
                            <span class="info-icon">ℹ️</span>
                            <span class="info-label">Info</span>
                        </button>
                    </div>
                    <p class="subtitle">Alege tipul de obiect de bucatarie, introdu parametrii, apoi genereaza predictia si scriptul Blender.</p>
                </div>
            </div>

            <div class="layout">
                <div class="panel input-panel">
                    <div class="panel-title">Input</div>
                    <form method="post" class="card">
                        <div class="grid">
                            <div>
                                <label for="object_type">object_type</label>
                                <select name="object_type" id="object_type" onchange="updateObjectTypeUI()">
                                    <option value="chair" {% if values.object_type == "chair" %}selected{% endif %}>chair</option>
                                    <option value="table" {% if values.object_type == "table" %}selected{% endif %}>table</option>
                                    <option value="cabinet" {% if values.object_type == "cabinet" %}selected{% endif %}>cabinet</option>
                                    <option value="fridge" {% if values.object_type == "fridge" %}selected{% endif %}>fridge</option>
                                    <option value="stove" {% if values.object_type == "stove" %}selected{% endif %}>stove</option>
                                </select>
                            </div>
                        </div>

                        <div class="section-title chair-only">Sezut</div>
                        <div class="grid chair-only">
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

                        <div class="section-title chair-only">Picioare</div>
                        <div class="grid chair-only">
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

                        <div class="section-title chair-only">Spatar</div>
                        <div class="grid chair-only">
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

                        <div class="section-title table-only">Masa</div>
                        <div class="grid table-only">
                            <div>
                                <label for="table_height">table_height</label>
                                <input type="number" step="0.01" min="0.35" max="0.9" name="table_height" id="table_height" value="{{ values.table_height }}" required />
                            </div>
                            <div>
                                <label for="table_width">table_width</label>
                                <input type="number" step="0.01" min="0.5" max="1.4" name="table_width" id="table_width" value="{{ values.table_width }}" required />
                            </div>
                            <div>
                                <label for="table_depth">table_depth</label>
                                <input type="number" step="0.01" min="0.5" max="1.0" name="table_depth" id="table_depth" value="{{ values.table_depth }}" required />
                            </div>
                            <div>
                                <label for="table_leg_count">leg_count</label>
                                <input type="number" step="1" min="3" max="4" name="table_leg_count" id="table_leg_count" value="{{ values.table_leg_count }}" required />
                            </div>
                            <div>
                                <label for="table_leg_thickness">leg_thickness</label>
                                <input type="number" step="0.01" min="0.04" max="0.09" name="table_leg_thickness" id="table_leg_thickness" value="{{ values.table_leg_thickness }}" required />
                            </div>
                            <div>
                                <label for="table_has_apron">has_apron</label>
                                <select name="table_has_apron" id="table_has_apron">
                                    <option value="0" {% if values.table_has_apron == 0 %}selected{% endif %}>0</option>
                                    <option value="1" {% if values.table_has_apron == 1 %}selected{% endif %}>1</option>
                                </select>
                            </div>
                            <div>
                                <label for="table_style_variant">style_variant</label>
                                <input type="number" step="1" min="0" max="2" name="table_style_variant" id="table_style_variant" value="{{ values.table_style_variant }}" required />
                            </div>
                        </div>

                        <div class="section-title cabinet-only">Dulap</div>
                        <div class="grid cabinet-only">
                            <div>
                                <label for="cabinet_height">cabinet_height</label>
                                <input type="number" step="0.01" min="1.0" max="2.2" name="cabinet_height" id="cabinet_height" value="{{ values.cabinet_height }}" required />
                            </div>
                            <div>
                                <label for="cabinet_width">cabinet_width</label>
                                <input type="number" step="0.01" min="0.6" max="1.6" name="cabinet_width" id="cabinet_width" value="{{ values.cabinet_width }}" required />
                            </div>
                            <div>
                                <label for="cabinet_depth">cabinet_depth</label>
                                <input type="number" step="0.01" min="0.3" max="0.8" name="cabinet_depth" id="cabinet_depth" value="{{ values.cabinet_depth }}" required />
                            </div>
                            <div>
                                <label for="wall_thickness">wall_thickness</label>
                                <input type="number" step="0.001" min="0.015" max="0.05" name="wall_thickness" id="wall_thickness" value="{{ values.wall_thickness }}" required />
                            </div>
                            <div>
                                <label for="door_type">door_type</label>
                                <select name="door_type" id="door_type">
                                    <option value="0" {% if values.door_type == 0 %}selected{% endif %}>flush_door</option>
                                    <option value="1" {% if values.door_type == 1 %}selected{% endif %}>inset_door</option>
                                </select>
                            </div>
                            <div>
                                <label for="door_count">door_count</label>
                                <input type="number" step="1" min="1" max="2" name="door_count" id="door_count" value="{{ values.door_count }}" required />
                            </div>
                            <div>
                                <label for="cabinet_style_variant">style_variant</label>
                                <input type="number" step="1" min="0" max="2" name="cabinet_style_variant" id="cabinet_style_variant" value="{{ values.cabinet_style_variant }}" required />
                            </div>
                        </div>

                        <div class="section-title fridge-only">Frigider</div>
                        <div class="grid fridge-only">
                            <div>
                                <label for="fridge_height">fridge_height</label>
                                <input type="number" step="0.01" min="1.4" max="2.1" name="fridge_height" id="fridge_height" value="{{ values.fridge_height }}" required />
                            </div>
                            <div>
                                <label for="fridge_width">fridge_width</label>
                                <input type="number" step="0.01" min="0.6" max="1.0" name="fridge_width" id="fridge_width" value="{{ values.fridge_width }}" required />
                            </div>
                            <div>
                                <label for="fridge_depth">fridge_depth</label>
                                <input type="number" step="0.01" min="0.55" max="0.8" name="fridge_depth" id="fridge_depth" value="{{ values.fridge_depth }}" required />
                            </div>
                            <div>
                                <label for="door_thickness">door_thickness</label>
                                <input type="number" step="0.01" min="0.02" max="0.05" name="door_thickness" id="door_thickness" value="{{ values.door_thickness }}" required />
                            </div>
                            <div>
                                <label for="fridge_handle_length">handle_length</label>
                                <input type="number" step="0.01" min="0.15" max="0.45" name="fridge_handle_length" id="fridge_handle_length" value="{{ values.fridge_handle_length }}" required />
                            </div>
                            <div>
                                <label for="freezer_ratio">freezer_ratio</label>
                                <input type="number" step="0.01" min="0.25" max="0.45" name="freezer_ratio" id="freezer_ratio" value="{{ values.freezer_ratio }}" required />
                            </div>
                            <div>
                                <label for="freezer_position">freezer_position</label>
                                <select name="freezer_position" id="freezer_position">
                                    <option value="0" {% if values.freezer_position == 0 %}selected{% endif %}>top_freezer</option>
                                    <option value="1" {% if values.freezer_position == 1 %}selected{% endif %}>bottom_freezer</option>
                                </select>
                            </div>
                            <div>
                                <label for="fridge_style_variant">style_variant</label>
                                <input type="number" step="1" min="0" max="2" name="fridge_style_variant" id="fridge_style_variant" value="{{ values.fridge_style_variant }}" required />
                            </div>
                        </div>

                        <div class="section-title stove-only">Aragaz</div>
                        <div class="grid stove-only">
                            <div>
                                <label for="stove_height">stove_height</label>
                                <input type="number" step="0.01" min="0.85" max="0.95" name="stove_height" id="stove_height" value="{{ values.stove_height }}" required />
                            </div>
                            <div>
                                <label for="stove_width">stove_width</label>
                                <input type="number" step="0.01" min="0.6" max="0.9" name="stove_width" id="stove_width" value="{{ values.stove_width }}" required />
                            </div>
                            <div>
                                <label for="stove_depth">stove_depth</label>
                                <input type="number" step="0.01" min="0.6" max="0.75" name="stove_depth" id="stove_depth" value="{{ values.stove_depth }}" required />
                            </div>
                            <div>
                                <label for="oven_height_ratio">oven_height_ratio</label>
                                <input type="number" step="0.01" min="0.45" max="0.6" name="oven_height_ratio" id="oven_height_ratio" value="{{ values.oven_height_ratio }}" required />
                            </div>
                            <div>
                                <label for="stove_handle_length">handle_length</label>
                                <input type="number" step="0.01" min="0.35" max="0.65" name="stove_handle_length" id="stove_handle_length" value="{{ values.stove_handle_length }}" required />
                            </div>
                            <div>
                                <label for="glass_thickness">glass_thickness</label>
                                <input type="number" step="0.01" min="0.01" max="0.03" name="glass_thickness" id="glass_thickness" value="{{ values.glass_thickness }}" required />
                            </div>
                            <div>
                                <label for="burner_count">burner_count</label>
                                <input type="number" step="1" min="4" max="4" name="burner_count" id="burner_count" value="{{ values.burner_count }}" required />
                            </div>
                            <div>
                                <label for="knob_count">knob_count</label>
                                <input type="number" step="1" min="6" max="6" name="knob_count" id="knob_count" value="{{ values.knob_count }}" required />
                            </div>
                            <div>
                                <label for="stove_style_variant">style_variant</label>
                                <input type="number" step="1" min="0" max="2" name="stove_style_variant" id="stove_style_variant" value="{{ values.stove_style_variant }}" required />
                            </div>
                        </div>

                        <div style="margin-top: 12px;">
                            <button type="submit">Predict</button>
                        </div>
                    </form>
                    <div class="status-board">
                        <div class="status-title">Status pornire</div>
                        <div class="status-item">
                            <span class="status-dot ok"></span>
                            <div class="status-text">
                                <div class="status-label">UI Flask</div>
                                <div class="status-meta">pornit</div>
                            </div>
                        </div>
                        <div class="status-item">
                            <span class="status-dot ok"></span>
                            <div class="status-text">
                                <div class="status-label">Model RN</div>
                                <div class="status-meta">incarcat</div>
                            </div>
                        </div>
                        <div class="status-item">
                            <span class="status-dot pending" id="status-blender-dot"></span>
                            <div class="status-text">
                                <div class="status-label">Blender API</div>
                                <div class="status-meta" id="status-blender-meta">verificare...</div>
                            </div>
                        </div>
                        <div class="status-note">Pentru preview randat, porneste Blender API.</div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-title">Output</div>
                    <div class="card output-card">
                        <div class="output-block">
                            <h3>Blender preview</h3>
                            <div class="preview" id="preview-container">
                                <img
                                    id="preview-image"
                                    src="{% if result and result.preview_src %}{{ result.preview_src }}{% endif %}"
                                    alt="Blender preview"
                                    style="{% if not (result and result.preview_src) %}display: none;{% endif %}"
                                />
                            </div>
                            <div class="preview-note" id="preview-note">
                                {% if result and result.preview_error %}
                                    Preview error: {{ result.preview_error }}
                                {% else %}
                                    Blender API pe 127.0.0.1:5001.
                                {% endif %}
                            </div>
                        </div>

                        <div class="output-block">
                            {% if result %}
                                <div class="result-title">Predicted variant: {{ result.label }}</div>
                                <div><strong>Confidence:</strong> {{ "%.2f" | format(result.confidence) }}</div>
                                <div><strong>Probabilities:</strong></div>
                                <pre class="code-block">{{ result.probabilities | tojson(indent=2) }}</pre>
                            {% else %}
                                <div class="result-title">Prediction</div>
                                <div class="preview-note">Completează datele și apasă Predict.</div>
                            {% endif %}
                        </div>

                        <div class="output-block" style="display: flex; flex-direction: column; gap: 10px; flex: 1;">
                            <div class="output-header">
                                <div><strong>Generated Blender script:</strong></div>
                            </div>
                            <div class="script-wrap">
                                <button class="copy-btn" type="button" onclick="copyScript()" aria-label="Copy script">
                                    <svg viewBox="0 0 24 24" role="img" aria-hidden="true">
                                        <path d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H10V7h9v14z"/>
                                    </svg>
                                </button>
                                <pre id="blender-script" class="code-block">{% if result %}{{ result.script }}{% endif %}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <button class="scroll-top" type="button" id="scroll-top" aria-label="Scroll to top">
                <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M12 4l7 7-1.41 1.41L13 7.83V20h-2V7.83L6.41 12.41 5 11z" />
                </svg>
            </button>

            <footer class="card footer">
                <div class="footer-line">Design & Development de <span class="footer-name">Denisa Elena Caldararu</span></div>
                <div class="footer-meta">© 2025-2026 • Built with Flask, TensorFlow/Keras, Blender API</div>
            </footer>

            <div class="info-overlay" id="infoOverlay" onclick="closeInfo(event)">
                <div class="info-modal">
                    <button class="info-close" onclick="closeInfo()" aria-label="Close">✕</button>

                    <h2>Despre proiect</h2>

                    <p>
                        Acest proiect demonstreaza un <strong>pipeline complet de inteligenta artificiala</strong>
                        care porneste de la date numerice, trece printr-o retea neuronala antrenata
                        si se finalizeaza cu generare procedurala de obiecte 3D in Blender.
                    </p>

                    <h3>Scopul proiectului</h3>
                    <p>
                        Scopul este clasificarea si interpretarea unor <strong>obiecte de mobilier</strong>
                        (scaune, mese, dulapuri) pe baza parametrilor geometrici,
                        urmata de generarea automata a unui model 3D corespunzator.
                    </p>

                    <h3>Logica retelei neuronale (RN)</h3>
                    <p>
                        Reteaua neuronala utilizata este o <strong>retea feed-forward (MLP)</strong>,
                        antrenata pe date tabulare numerice.
                        RN-ul <em>nu genereaza geometrie</em>, ci invata sa recunoasca
                        <strong>tipul obiectului / varianta constructiva</strong>
                        pe baza relatiilor dintre dimensiuni.
                    </p>

                    <ul>
                        <li>Intrare: dimensiuni si parametri (ex: inaltime, latime, grosimi, optiuni)</li>
                        <li>Iesire: clasa discreta (ex: scaun simplu, masa joasa, dulap cu doua usi)</li>
                    </ul>

                    <p>
                        Deciziile RN-ului sunt apoi folosite pentru a selecta
                        <strong>reguli deterministe de generare Blender</strong>.
                    </p>

                    <h3>Generarea setului de date</h3>
                    <p>
                        Seturile de date sunt <strong>sintetice</strong> si sunt generate programatic,
                        folosind intervale realiste si reguli logice.
                    </p>

                    <ul>
                        <li>Intervalele sunt alese din ergonomie si design real</li>
                        <li>Reguli logice elimina combinatii imposibile (ex: spatar fara inaltime)</li>
                        <li>Etichetele sunt atribuite determinist, nu manual</li>
                    </ul>

                    <p>
                        Aceasta abordare permite:
                    </p>
                    <ul>
                        <li>control total asupra distributiei datelor</li>
                        <li>reproductibilitate</li>
                        <li>explicabilitate clara a deciziilor RN</li>
                    </ul>

                    <h3>Legatura RN → Blender</h3>
                    <p>
                        RN-ul nu „deseneaza”.
                        El decide <strong>ce este obiectul</strong>.
                        Scripturile Blender folosesc parametrii initiali + clasa prezisa
                        pentru a construi un model 3D coerent si realist.
                    </p>

                    <h3>Design si arhitectura</h3>
                    <ul>
                        <li>RN separata pentru fiecare tip de obiect</li>
                        <li>Pipeline clar: generate → clean → scale → split → train → infer</li>
                        <li>UI decupleaza complet logica RN de logica Blender</li>
                    </ul>

                    <h3>Context si lucrari similare</h3>
                    <p>
                        Proiectul se inspira din concepte utilizate in:
                    </p>
                    <ul>
                        <li>procedural modeling (Blender, Houdini)</li>
                        <li>design parametric asistat de AI</li>
                        <li>mass customization si generative design</li>
                    </ul>

                    <p>
                        Diferenta majora este integrarea completa:
                        <strong>RN explicabil → generare 3D determinista</strong>.
                    </p>
                </div>
            </div>

        <script>
            function openInfo() {
                const overlay = document.getElementById('infoOverlay');
                if (!overlay) {
                    return;
                }
                overlay.classList.add('show');
                document.body.style.overflow = 'hidden';
            }

            function closeInfo(event) {
                const overlay = document.getElementById('infoOverlay');
                if (!overlay) {
                    return;
                }
                if (!event || event.target.id === 'infoOverlay') {
                    overlay.classList.remove('show');
                    document.body.style.overflow = '';
                }
            }

            const hasBackrest = document.getElementById('has_backrest');
            const backrestHeight = document.getElementById('backrest_height');
            const objectTypeSelect = document.getElementById('object_type');
            const scrollTopButton = document.getElementById('scroll-top');
            const blenderDot = document.getElementById('status-blender-dot');
            const blenderMeta = document.getElementById('status-blender-meta');

            function updateBlenderStatus(status) {
                if (!blenderDot || !blenderMeta) {
                    return;
                }
                if (status && status.blender_api_ok) {
                    blenderDot.classList.remove('pending', 'down');
                    blenderDot.classList.add('ok');
                    blenderMeta.textContent = 'conectat';
                } else {
                    blenderDot.classList.remove('pending', 'ok');
                    blenderDot.classList.add('down');
                    blenderMeta.textContent = 'nepornit';
                }
            }

            function syncBackrest() {
                if (!hasBackrest || !backrestHeight) {
                    return;
                }
                if (objectTypeSelect && objectTypeSelect.value !== 'chair') {
                    backrestHeight.setAttribute('readonly', 'readonly');
                    return;
                }
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

            if (hasBackrest) {
                hasBackrest.addEventListener('change', syncBackrest);
            }

            function updateObjectTypeUI() {
                const chairBlocks = document.querySelectorAll('.chair-only');
                const tableBlocks = document.querySelectorAll('.table-only');
                const cabinetBlocks = document.querySelectorAll('.cabinet-only');
                const fridgeBlocks = document.querySelectorAll('.fridge-only');
                const stoveBlocks = document.querySelectorAll('.stove-only');
                const objectType = objectTypeSelect ? objectTypeSelect.value : 'chair';

                chairBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'chair'));
                tableBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'table'));
                cabinetBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'cabinet'));
                fridgeBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'fridge'));
                stoveBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'stove'));

                const toggleInputs = (blocks, enabled) => {
                    blocks.forEach((node) => {
                        node.querySelectorAll('input, select').forEach((field) => {
                            if (enabled) {
                                field.removeAttribute('disabled');
                                if (field.dataset.wasRequired === 'true') {
                                    field.setAttribute('required', 'required');
                                    delete field.dataset.wasRequired;
                                }
                            } else {
                                field.setAttribute('disabled', 'disabled');
                                if (field.hasAttribute('required')) {
                                    field.dataset.wasRequired = 'true';
                                    field.removeAttribute('required');
                                }
                            }
                        });
                    });
                };

                toggleInputs(chairBlocks, objectType === 'chair');
                toggleInputs(tableBlocks, objectType === 'table');
                toggleInputs(cabinetBlocks, objectType === 'cabinet');
                toggleInputs(fridgeBlocks, objectType === 'fridge');
                toggleInputs(stoveBlocks, objectType === 'stove');

                syncBackrest();
            }

            if (objectTypeSelect) {
                objectTypeSelect.addEventListener('change', updateObjectTypeUI);
            }
            updateObjectTypeUI();

            function copyScript() {
                const script = document.getElementById('blender-script');
                if (!script) {
                    return;
                }
                navigator.clipboard.writeText(script.textContent || '')
                    .then(() => alert('Script copied to clipboard.'))
                    .catch(() => alert('Copy failed.'));
            }


            function updateScrollTopButton() {
                if (!scrollTopButton) {
                    return;
                }
                const scrollPosition = window.pageYOffset
                    || document.documentElement.scrollTop
                    || document.body.scrollTop
                    || 0;
                if (scrollPosition > 220) {
                    scrollTopButton.classList.add('show');
                } else {
                    scrollTopButton.classList.remove('show');
                }
            }

            function scrollToTopImmediate() {
                const topAnchor = document.getElementById('top');
                if (topAnchor) {
                    topAnchor.scrollIntoView({ behavior: 'auto' });
                }
                window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
                document.documentElement.scrollTop = 0;
                document.body.scrollTop = 0;
            }

            if ('scrollRestoration' in history) {
                history.scrollRestoration = 'manual';
            }

            window.addEventListener('scroll', updateScrollTopButton, { passive: true });
            window.addEventListener('load', () => {
                scrollToTopImmediate();
                setTimeout(scrollToTopImmediate, 0);
                updateScrollTopButton();
                fetch('/status')
                    .then((response) => response.json())
                    .then((data) => updateBlenderStatus(data))
                    .catch(() => updateBlenderStatus({ blender_api_ok: false }));
            });
            window.addEventListener('pageshow', (event) => {
                if (event.persisted) {
                    scrollToTopImmediate();
                }
            });

            window.addEventListener('beforeunload', () => {
                document.documentElement.scrollTop = 0;
                document.body.scrollTop = 0;
            });

            if (scrollTopButton) {
                scrollTopButton.addEventListener('click', () => {
                    window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
                });
            }
        </script>
        </div>
    </body>
</html>
"""


def parse_float(name: str, default: float) -> float:
    # Parseaza float din formular, cu default daca lipseste.
    value = request.form.get(name, default)
    return float(value)


def parse_int(name: str, default: int) -> int:
    # Parseaza int din formular, cu default daca lipseste.
    value = request.form.get(name, default)
    return int(value)


def parse_str(name: str, default: str) -> str:
    # Parseaza string din formular, cu default daca lipseste.
    value = request.form.get(name, default)
    return str(value)


def request_preview(payload: dict) -> tuple[str | None, str | None]:
    # Trimite payload la Blender API si returneaza imaginea base64 sau eroare.
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
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
            details = body.get("details")
            if details:
                return None, f"{exc}: {details}"
            return None, f"{exc}: {body.get('error', 'Unknown error')}"
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, str(exc)
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        return None, str(exc)


@app.route("/", methods=["GET", "POST"])
def index():
    # Valori default pentru primul render si fallback-uri.
    values = {
        "object_type": "chair",
        "seat_height": 0.55,
        "seat_width": 0.45,
        "seat_depth": 0.45,
        "leg_count": 4,
        "leg_shape": "square",
        "leg_size": 0.05,
        "has_backrest": 1,
        "backrest_height": 0.25,
        "style_variant": 0,
        "table_height": 0.75,
        "table_width": 1.0,
        "table_depth": 0.7,
        "table_leg_count": 4,
        "table_leg_thickness": 0.06,
        "table_has_apron": 1,
        "table_style_variant": 0,
        "cabinet_height": 1.8,
        "cabinet_width": 1.0,
        "cabinet_depth": 0.5,
        "wall_thickness": 0.03,
        "door_type": 0,
        "door_count": 2,
        "cabinet_style_variant": 0,
        "fridge_height": 1.8,
        "fridge_width": 0.8,
        "fridge_depth": 0.65,
        "door_thickness": 0.03,
        "fridge_handle_length": 0.3,
        "freezer_ratio": 0.35,
        "freezer_position": 0,
        "fridge_style_variant": 0,
        "stove_height": 0.9,
        "stove_width": 0.7,
        "stove_depth": 0.65,
        "oven_height_ratio": 0.5,
        "stove_handle_length": 0.5,
        "glass_thickness": 0.02,
        "burner_count": 4,
        "knob_count": 6,
        "stove_style_variant": 0,
        "rotate_yaw": 35.0,
        "rotate_pitch": 15.0,
    }

    # Payload final pentru randarea template-ului.
    result = None
    if request.method == "POST":
        # Stabileste tipul obiectului pentru a sti ce campuri se parseaza.
        values["object_type"] = parse_str("object_type", values["object_type"])
        object_type = values["object_type"] if values["object_type"] in LABEL_MAPS else "chair"
        values["object_type"] = object_type

        if object_type == "table":
            # Parseaza inputurile table, ruleaza inferenta si genereaza script Blender.
            values["table_height"] = parse_float("table_height", values["table_height"])
            values["table_width"] = parse_float("table_width", values["table_width"])
            values["table_depth"] = parse_float("table_depth", values["table_depth"])
            values["table_leg_count"] = parse_int("table_leg_count", values["table_leg_count"])
            values["table_leg_thickness"] = parse_float("table_leg_thickness", values["table_leg_thickness"])
            values["table_has_apron"] = parse_int("table_has_apron", values["table_has_apron"])
            values["table_style_variant"] = parse_int("table_style_variant", values["table_style_variant"])
            values["rotate_yaw"] = parse_float("rotate_yaw", values["rotate_yaw"])
            values["rotate_pitch"] = parse_float("rotate_pitch", values["rotate_pitch"])

            model, scaler = load_artifacts("table")
            features = build_input_array("table", values)
            scaled_features = scaler.transform(features)
            probabilities = model.predict(scaled_features, verbose=0)[0]
            predicted_label = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

            script = generate_table_script(
                table_height=values["table_height"],
                table_width=values["table_width"],
                table_depth=values["table_depth"],
                leg_count=values["table_leg_count"],
                leg_thickness=values["table_leg_thickness"],
                has_apron=values["table_has_apron"],
                style_variant=values["table_style_variant"],
            )

            preview_payload = {
                "object_type": "table",
                "table_height": values["table_height"],
                "table_width": values["table_width"],
                "table_depth": values["table_depth"],
                "leg_count": values["table_leg_count"],
                "leg_thickness": values["table_leg_thickness"],
                "has_apron": values["table_has_apron"],
                "style_variant": values["table_style_variant"],
                "rotate_yaw": values["rotate_yaw"],
                "rotate_pitch": values["rotate_pitch"],
            }
            preview_image, preview_error = request_preview(preview_payload)
            preview_src = f"data:image/png;base64,{preview_image}" if preview_image else None
            label_map = LABEL_MAPS["table"]
        elif object_type == "cabinet":
            # Parseaza inputurile cabinet, ruleaza inferenta si genereaza script Blender.
            values["cabinet_height"] = parse_float("cabinet_height", values["cabinet_height"])
            values["cabinet_width"] = parse_float("cabinet_width", values["cabinet_width"])
            values["cabinet_depth"] = parse_float("cabinet_depth", values["cabinet_depth"])
            values["wall_thickness"] = parse_float("wall_thickness", values["wall_thickness"])
            values["door_type"] = parse_int("door_type", values["door_type"])
            values["door_count"] = parse_int("door_count", values["door_count"])
            values["cabinet_style_variant"] = parse_int("cabinet_style_variant", values["cabinet_style_variant"])
            values["rotate_yaw"] = parse_float("rotate_yaw", values["rotate_yaw"])
            values["rotate_pitch"] = parse_float("rotate_pitch", values["rotate_pitch"])

            model, scaler = load_artifacts("cabinet")
            features = build_input_array("cabinet", values)
            scaled_features = scaler.transform(features)
            probabilities = model.predict(scaled_features, verbose=0)[0]
            predicted_label = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

            script = generate_cabinet_script(
                cabinet_height=values["cabinet_height"],
                cabinet_width=values["cabinet_width"],
                cabinet_depth=values["cabinet_depth"],
                wall_thickness=values["wall_thickness"],
                door_type=values["door_type"],
                door_count=values["door_count"],
                style_variant=values["cabinet_style_variant"],
            )

            preview_payload = {
                "object_type": "cabinet",
                "cabinet_height": values["cabinet_height"],
                "cabinet_width": values["cabinet_width"],
                "cabinet_depth": values["cabinet_depth"],
                "wall_thickness": values["wall_thickness"],
                "door_type": values["door_type"],
                "door_count": values["door_count"],
                "style_variant": values["cabinet_style_variant"],
                "rotate_yaw": values["rotate_yaw"],
                "rotate_pitch": values["rotate_pitch"],
            }
            preview_image, preview_error = request_preview(preview_payload)
            preview_src = f"data:image/png;base64,{preview_image}" if preview_image else None
            label_map = LABEL_MAPS["cabinet"]
        elif object_type == "fridge":
            # Parseaza inputurile fridge, ruleaza inferenta si genereaza script Blender.
            values["fridge_height"] = parse_float("fridge_height", values["fridge_height"])
            values["fridge_width"] = parse_float("fridge_width", values["fridge_width"])
            values["fridge_depth"] = parse_float("fridge_depth", values["fridge_depth"])
            values["door_thickness"] = parse_float("door_thickness", values["door_thickness"])
            values["fridge_handle_length"] = parse_float("fridge_handle_length", values["fridge_handle_length"])
            values["freezer_ratio"] = parse_float("freezer_ratio", values["freezer_ratio"])
            values["freezer_position"] = parse_int("freezer_position", values["freezer_position"])
            values["fridge_style_variant"] = parse_int("fridge_style_variant", values["fridge_style_variant"])
            values["rotate_yaw"] = parse_float("rotate_yaw", values["rotate_yaw"])
            values["rotate_pitch"] = parse_float("rotate_pitch", values["rotate_pitch"])

            model, scaler = load_artifacts("fridge")
            features = build_input_array("fridge", values)
            scaled_features = scaler.transform(features)
            probabilities = model.predict(scaled_features, verbose=0)[0]
            predicted_label = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

            script = generate_fridge_script(
                fridge_height=values["fridge_height"],
                fridge_width=values["fridge_width"],
                fridge_depth=values["fridge_depth"],
                door_thickness=values["door_thickness"],
                handle_length=values["fridge_handle_length"],
                freezer_ratio=values["freezer_ratio"],
                freezer_position=values["freezer_position"],
                style_variant=values["fridge_style_variant"],
            )

            preview_payload = {
                "object_type": "fridge",
                "fridge_height": values["fridge_height"],
                "fridge_width": values["fridge_width"],
                "fridge_depth": values["fridge_depth"],
                "door_thickness": values["door_thickness"],
                "handle_length": values["fridge_handle_length"],
                "freezer_ratio": values["freezer_ratio"],
                "freezer_position": values["freezer_position"],
                "style_variant": values["fridge_style_variant"],
                "rotate_yaw": values["rotate_yaw"],
                "rotate_pitch": values["rotate_pitch"],
            }
            preview_image, preview_error = request_preview(preview_payload)
            preview_src = f"data:image/png;base64,{preview_image}" if preview_image else None
            label_map = LABEL_MAPS["fridge"]
        elif object_type == "stove":
            # Parseaza inputurile stove, ruleaza inferenta si genereaza script Blender.
            values["stove_height"] = parse_float("stove_height", values["stove_height"])
            values["stove_width"] = parse_float("stove_width", values["stove_width"])
            values["stove_depth"] = parse_float("stove_depth", values["stove_depth"])
            values["oven_height_ratio"] = parse_float("oven_height_ratio", values["oven_height_ratio"])
            values["stove_handle_length"] = parse_float("stove_handle_length", values["stove_handle_length"])
            values["glass_thickness"] = parse_float("glass_thickness", values["glass_thickness"])
            values["burner_count"] = parse_int("burner_count", values["burner_count"])
            values["knob_count"] = parse_int("knob_count", values["knob_count"])
            values["stove_style_variant"] = parse_int("stove_style_variant", values["stove_style_variant"])
            values["rotate_yaw"] = parse_float("rotate_yaw", values["rotate_yaw"])
            values["rotate_pitch"] = parse_float("rotate_pitch", values["rotate_pitch"])

            model, scaler = load_artifacts("stove")
            features = build_input_array("stove", values)
            scaled_features = scaler.transform(features)
            probabilities = model.predict(scaled_features, verbose=0)[0]
            predicted_label = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

            script = generate_stove_script(
                stove_height=values["stove_height"],
                stove_width=values["stove_width"],
                stove_depth=values["stove_depth"],
                oven_height_ratio=values["oven_height_ratio"],
                handle_length=values["stove_handle_length"],
                glass_thickness=values["glass_thickness"],
                style_variant=values["stove_style_variant"],
            )

            preview_payload = {
                "object_type": "stove",
                "stove_height": values["stove_height"],
                "stove_width": values["stove_width"],
                "stove_depth": values["stove_depth"],
                "oven_height_ratio": values["oven_height_ratio"],
                "handle_length": values["stove_handle_length"],
                "glass_thickness": values["glass_thickness"],
                "style_variant": values["stove_style_variant"],
                "rotate_yaw": values["rotate_yaw"],
                "rotate_pitch": values["rotate_pitch"],
            }
            preview_image, preview_error = request_preview(preview_payload)
            preview_src = f"data:image/png;base64,{preview_image}" if preview_image else None
            label_map = LABEL_MAPS["stove"]
        else:
            # Parseaza inputurile chair, ruleaza inferenta si genereaza script Blender.
            values["seat_height"] = parse_float("seat_height", values["seat_height"])
            values["seat_width"] = parse_float("seat_width", values["seat_width"])
            values["seat_depth"] = parse_float("seat_depth", values["seat_depth"])
            values["leg_count"] = parse_int("leg_count", values["leg_count"])
            values["leg_shape"] = parse_str("leg_shape", values["leg_shape"])
            values["leg_size"] = parse_float("leg_size", values["leg_size"])
            values["has_backrest"] = parse_int("has_backrest", values["has_backrest"])
            values["backrest_height"] = parse_float("backrest_height", values["backrest_height"])
            values["style_variant"] = parse_int("style_variant", values["style_variant"])
            values["rotate_yaw"] = parse_float("rotate_yaw", values["rotate_yaw"])
            values["rotate_pitch"] = parse_float("rotate_pitch", values["rotate_pitch"])

            if values["has_backrest"] == 0:
                values["backrest_height"] = 0.0
            else:
                values["backrest_height"] = max(0.2, min(0.5, values["backrest_height"]))

            model, scaler = load_artifacts("chair")
            features = build_input_array("chair", values)
            scaled_features = scaler.transform(features)
            probabilities = model.predict(scaled_features, verbose=0)[0]
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
                "object_type": "chair",
                "seat_height": values["seat_height"],
                "seat_width": values["seat_width"],
                "seat_depth": values["seat_depth"],
                "leg_count": values["leg_count"],
                "leg_shape": values["leg_shape"],
                "leg_size": values["leg_size"],
                "has_backrest": values["has_backrest"],
                "backrest_height": values["backrest_height"],
                "style_variant": values["style_variant"],
                "rotate_yaw": values["rotate_yaw"],
                "rotate_pitch": values["rotate_pitch"],
            }
            preview_image, preview_error = request_preview(preview_payload)
            preview_src = f"data:image/png;base64,{preview_image}" if preview_image else None
            label_map = LABEL_MAPS["chair"]

        result = {
            "label": label_map[predicted_label],
            "confidence": confidence,
            "probabilities": {label_map[idx]: float(prob) for idx, prob in enumerate(probabilities)},
            "script": script,
            "preview_src": preview_src,
            "preview_error": preview_error,
        }

    return render_template_string(
        HTML_TEMPLATE,
        values=values,
        result=result,
        blender_api_url=BLENDER_API_URL,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
