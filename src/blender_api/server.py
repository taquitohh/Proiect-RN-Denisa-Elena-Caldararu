from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blender_scripts.chair_generator import generate_chair_script


BLENDER_PATH = os.environ.get("BLENDER_PATH", "blender")
OUTPUT_DIR = Path("results") / "renders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def build_render_script(params: dict[str, Any], output_path: Path) -> str:
    chair_script = generate_chair_script(
        seat_height=float(params["seat_height"]),
        seat_width=float(params["seat_width"]),
        seat_depth=float(params["seat_depth"]),
        leg_count=int(params["leg_count"]),
        leg_shape=str(params["leg_shape"]),
        leg_size=float(params["leg_size"]),
        has_backrest=int(params["has_backrest"]),
        backrest_height=float(params["backrest_height"]),
        style_variant=int(params["style_variant"]),
    )

    render_script = f"""
{chair_script}

# -------------------------
# CAMERA + LIGHT
# -------------------------
import bpy

bpy.ops.object.camera_add(location=(2.6, -2.6, 2.2))
camera = bpy.context.active_object
camera.rotation_euler = (1.1, 0.0, 0.9)

bpy.ops.object.light_add(type='AREA', location=(2.0, -1.5, 3.0))
light = bpy.context.active_object
light.data.energy = 500

bpy.context.scene.camera = camera
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 600
bpy.context.scene.render.filepath = r"{output_path.as_posix()}"

bpy.ops.render.render(write_still=True)
"""

    return render_script.strip() + "\n"


def run_blender(script_path: Path) -> None:
    cmd = [BLENDER_PATH, "-b", "-P", str(script_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@app.route("/render", methods=["POST"])
def render_preview():
    params = request.get_json(silent=True) or {}
    required = {
        "seat_height",
        "seat_width",
        "seat_depth",
        "leg_count",
        "leg_shape",
        "leg_size",
        "has_backrest",
        "backrest_height",
        "style_variant",
    }
    missing = sorted(required - set(params))
    if missing:
        return jsonify({"error": f"Missing params: {', '.join(missing)}"}), 400

    output_path = OUTPUT_DIR / "chair_preview.png"
    script_text = build_render_script(params, output_path)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script_text)
        script_path = Path(tmp.name)

    try:
        run_blender(script_path)
    except subprocess.CalledProcessError as exc:
        return jsonify({"error": "Blender render failed", "details": exc.stderr}), 500
    finally:
        if script_path.exists():
            script_path.unlink(missing_ok=True)

    if not output_path.exists():
        return jsonify({"error": "Render not produced"}), 500

    with output_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")

    return jsonify({"image_base64": encoded})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
