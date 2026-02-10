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

# Permite rularea API-ului din orice director prin setarea sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blender_scripts.chair_generator import generate_chair_script
from src.blender_scripts.table_generator import generate_table_script
from src.blender_scripts.cabinet_generator import generate_cabinet_script
from src.blender_scripts.fridge_generator import generate_fridge_script
from src.blender_scripts.stove_generator import generate_stove_script


def resolve_blender_path() -> str:
    # Prefera BLENDER_PATH explicit, apoi cauta in locatii uzuale.
    env_path = os.environ.get("BLENDER_PATH")
    if env_path:
        return env_path

    candidates = [
        "C:/Program Files/Blender Foundation/blender.exe",
        "C:/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe",
        "D:/SteamLibrary/steamapps/common/Blender/blender.exe",
        "E:/SteamLibrary/steamapps/common/Blender/blender.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    # Fallback: incearca executabilul din PATH.
    return "blender"


# Rezolva calea catre Blender si creeaza folderul pentru randari.
BLENDER_PATH = resolve_blender_path()
OUTPUT_DIR = Path("results") / "renders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
print(f"Blender API using BLENDER_PATH={BLENDER_PATH}")


@app.after_request
def add_cors_headers(response):
    # Permite UI-ului local sa apeleze API-ul din alt origin.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def build_render_script(params: dict[str, Any], output_path: Path) -> str:
    # Alege generatorul in functie de tipul obiectului si mapeaza campurile.
    object_type = str(params.get("object_type", "chair"))
    if object_type == "table":
        object_script = generate_table_script(
            table_height=float(params["table_height"]),
            table_width=float(params["table_width"]),
            table_depth=float(params["table_depth"]),
            leg_count=int(params["leg_count"]),
            leg_thickness=float(params["leg_thickness"]),
            has_apron=int(params["has_apron"]),
            style_variant=int(params["style_variant"]),
        )
    elif object_type == "cabinet":
        object_script = generate_cabinet_script(
            cabinet_height=float(params["cabinet_height"]),
            cabinet_width=float(params["cabinet_width"]),
            cabinet_depth=float(params["cabinet_depth"]),
            wall_thickness=float(params["wall_thickness"]),
            door_type=int(params["door_type"]),
            door_count=int(params["door_count"]),
            style_variant=int(params["style_variant"]),
        )
    elif object_type == "fridge":
        object_script = generate_fridge_script(
            fridge_height=float(params["fridge_height"]),
            fridge_width=float(params["fridge_width"]),
            fridge_depth=float(params["fridge_depth"]),
            door_thickness=float(params["door_thickness"]),
            handle_length=float(params["handle_length"]),
            freezer_ratio=float(params["freezer_ratio"]),
            freezer_position=int(params["freezer_position"]),
            style_variant=int(params["style_variant"]),
        )
    elif object_type == "stove":
        object_script = generate_stove_script(
            stove_height=float(params["stove_height"]),
            stove_width=float(params["stove_width"]),
            stove_depth=float(params["stove_depth"]),
            oven_height_ratio=float(params["oven_height_ratio"]),
            handle_length=float(params["handle_length"]),
            glass_thickness=float(params["glass_thickness"]),
            style_variant=int(params["style_variant"]),
        )
    else:
        object_script = generate_chair_script(
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

    # Aplica rotatii default si ajustari specifice obiectului.
    rotate_yaw = float(params.get("rotate_yaw", 35.0))
    rotate_pitch = float(params.get("rotate_pitch", 15.0))
    if object_type == "cabinet":
        rotate_yaw = (rotate_yaw + 180.0) % 360.0
    if object_type == "fridge":
        rotate_yaw = (rotate_yaw + 180.0) % 360.0

    # Construieste scriptul Blender cu camera, lumina si setari de randare.
    render_script = f"""
{object_script}

# -------------------------
# CAMERA + LUMINA
# -------------------------
import bpy
import math

# Centreaza obiectul in origine si aplica rotatia
if bpy.context.selected_objects:
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.location_clear()
    bpy.ops.object.rotation_clear()
    bpy.ops.object.rotation_euler = (math.radians({rotate_pitch}), 0.0, math.radians({rotate_yaw}))

bpy.ops.object.camera_add(location=(2.6, -2.6, 2.2))
camera = bpy.context.active_object

# Asigura ca camera priveste spre origine pentru centrare consistenta
constraint = camera.constraints.new(type='TRACK_TO')
constraint.target = bpy.data.objects.new("CameraTarget", None)
bpy.context.scene.collection.objects.link(constraint.target)
target_offset_x = -0.12
constraint.target.location = (target_offset_x, 0.0, 0.0)
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

bpy.ops.object.light_add(type='AREA', location=(2.0, -1.5, 3.0))
light = bpy.context.active_object
light.data.energy = 500

bpy.context.scene.camera = camera
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 600
bpy.context.scene.render.filepath = r"{output_path.resolve().as_posix()}"

bpy.ops.render.render(write_still=True)
"""

    return render_script.strip() + "\n"


def run_blender(script_path: Path) -> subprocess.CompletedProcess[str]:
    # Ruleaza Blender headless cu scriptul generat.
    cmd = [BLENDER_PATH, "-b", "-P", str(script_path)]
    if not Path(BLENDER_PATH).exists() and BLENDER_PATH != "blender":
        raise FileNotFoundError(f"BLENDER_PATH not found: {BLENDER_PATH}")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


@app.route("/render", methods=["POST"])
def render_preview():
    # Valideaza payload-ul, ruleaza Blender si returneaza imaginea base64.
    params = request.get_json(silent=True) or {}
    object_type = str(params.get("object_type", "chair"))
    if object_type == "table":
        required = {
            "table_height",
            "table_width",
            "table_depth",
            "leg_count",
            "leg_thickness",
            "has_apron",
            "style_variant",
        }
    elif object_type == "cabinet":
        required = {
            "cabinet_height",
            "cabinet_width",
            "cabinet_depth",
            "wall_thickness",
            "door_type",
            "door_count",
            "style_variant",
        }
    elif object_type == "fridge":
        required = {
            "fridge_height",
            "fridge_width",
            "fridge_depth",
            "door_thickness",
            "handle_length",
            "freezer_ratio",
            "freezer_position",
            "style_variant",
        }
    elif object_type == "stove":
        required = {
            "stove_height",
            "stove_width",
            "stove_depth",
            "oven_height_ratio",
            "handle_length",
            "glass_thickness",
            "style_variant",
        }
    else:
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

    # Foloseste un nume fix pentru imaginea de preview.
    output_path = OUTPUT_DIR / "chair_preview.png"
    script_text = build_render_script(params, output_path)

    # Scrie scriptul Blender temporar pe disk.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script_text)
        script_path = Path(tmp.name)

    try:
        result = run_blender(script_path)
        if result.returncode != 0:
            details = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            print(f"Blender render failed: {details}", flush=True)
            return jsonify({"error": "Blender render failed", "details": details}), 500
    except FileNotFoundError as exc:
        print(f"Blender executable not found: {exc}", flush=True)
        return jsonify({"error": "Blender executable not found", "details": str(exc)}), 500
    except Exception as exc:
        print(f"Unexpected render error: {exc}", flush=True)
        return jsonify({"error": "Unexpected render error", "details": str(exc)}), 500
    finally:
        if script_path.exists():
            script_path.unlink(missing_ok=True)

    if not output_path.exists():
        details = {
            "cwd": str(Path.cwd()),
            "expected_output": str(output_path.resolve()),
        }
        print(f"Render not produced: {details}", flush=True)
        return jsonify({"error": "Render not produced", "details": details}), 500

    # Returneaza imaginea randata ca base64 pentru UI.
    with output_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")

    return jsonify({"image_base64": encoded})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
