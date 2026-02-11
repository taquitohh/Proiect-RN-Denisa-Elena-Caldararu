# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

from __future__ import annotations
import textwrap


def generate_chair_script(
    seat_height: float,        # inaltime totala de la podea la sezut
    seat_width: float,
    seat_depth: float,
    leg_count: int,
    leg_shape: str,
    leg_size: float,
    has_backrest: int,
    backrest_height: float,
    style_variant: int,
) -> str:

    # Limiteaza inputurile pentru a pastra geometria stabila si predictibila.

    # -------------------------
    # SIGURANTA / LIMITARI
    # -------------------------
    leg_count = max(3, min(5, int(leg_count)))
    has_backrest = 1 if int(has_backrest) == 1 else 0
    backrest_height = max(0.0, float(backrest_height))

    style_offset = {0: 0.0, 1: 0.02, 2: -0.02}.get(style_variant, 0.0)

    # Construieste un script Blender cu pasi expliciti de geometrie.
    script = f"""
import bpy

# -------------------------
# PARAMETRI
# -------------------------
seat_height = {seat_height}
seat_width = {seat_width}
seat_depth = {seat_depth}
leg_count = {leg_count}
leg_shape = "{leg_shape}"
leg_size = {leg_size}
has_backrest = {has_backrest}
backrest_height = {backrest_height}
style_offset = {style_offset}

# -------------------------
# CURATA SCENA (porneste de la o scena goala)
# -------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# -------------------------
# DERIVED DIMENSIONS (calculate din inputuri)
# -------------------------
seat_thickness = max(0.03, seat_height * 0.2)
leg_height = max(0.01, seat_height - seat_thickness)

leg_contact = leg_size if leg_shape == "round" else (leg_size / 2.0)

seat_bottom_z = leg_height
seat_top_z = seat_bottom_z + seat_thickness

# -------------------------
# SEZUT (suprafata principala)
# -------------------------
bpy.ops.mesh.primitive_cube_add(size=2)
seat = bpy.context.active_object
seat.name = "Seat"
seat.scale = (
    seat_width / 2.0,
    seat_depth / 2.0,
    seat_thickness / 2.0
)
seat.location = (
    0.0,
    0.0,
    seat_bottom_z + seat_thickness / 2.0
)

# -------------------------
# POZITII PICIOARE (contact exact cu marginea)
# -------------------------
x_offset = seat_width / 2.0 - leg_contact + style_offset
y_offset = seat_depth / 2.0 - leg_contact + style_offset

leg_positions = [
    ( x_offset,  y_offset),
    (-x_offset,  y_offset),
    ( x_offset, -y_offset),
    (-x_offset, -y_offset),
][:leg_count]

legs = []

# -------------------------
# LEGS (rotunde sau patrate, contact exact)
# -------------------------
for idx, (x, y) in enumerate(leg_positions, start=1):
    if leg_shape == "round":
        bpy.ops.mesh.primitive_cylinder_add(
            radius=leg_size,
            depth=leg_height
        )
    else:
        bpy.ops.mesh.primitive_cube_add(size=2)
        bpy.context.active_object.scale = (
            leg_size / 2.0,
            leg_size / 2.0,
            leg_height / 2.0
        )
    leg = bpy.context.active_object
    leg.name = f"Leg_{{idx}}"
    leg.location = (
        x,
        y,
        leg_height / 2.0
    )
    legs.append(leg)

# -------------------------
# SPATAR (atinge sezutul exact)
# -------------------------
if has_backrest == 1 and backrest_height > 0:
    backrest_thickness = leg_size

    bpy.ops.mesh.primitive_cube_add(size=2)
    backrest = bpy.context.active_object
    backrest.name = "Backrest"
    backrest.scale = (
        seat_width / 2.0,
        backrest_thickness / 2.0,
        backrest_height / 2.0
    )
    backrest.location = (
        0.0,
        -(seat_depth / 2.0) + backrest_thickness / 2.0,
        seat_top_z + backrest_height / 2.0
    )

# -------------------------
# OPTIONAL: JOIN ALL PARTS (o singura plasa)
# -------------------------
join_objects = True
if join_objects:
    bpy.ops.object.select_all(action='DESELECT')
    seat.select_set(True)
    for leg in legs:
        leg.select_set(True)
    if has_backrest == 1 and backrest_height > 0:
        backrest.select_set(True)

    bpy.context.view_layer.objects.active = seat
    bpy.ops.object.join()
"""

    return textwrap.dedent(script).strip()
