# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

from __future__ import annotations

import textwrap


def generate_table_script(
    table_height: float,
    table_width: float,
    table_depth: float,
    leg_count: int,
    leg_thickness: float,
    has_apron: int,
    style_variant: int,
) -> str:
    # Limiteaza inputurile si aplica un mic offset de stil.
    leg_count = max(3, min(4, int(leg_count)))
    has_apron = 1 if int(has_apron) == 1 else 0
    style_adjust = {0: 0.0, 1: 0.02, 2: -0.02}.get(int(style_variant), 0.0)

    # Construieste un script Blender cu pasi expliciti de geometrie.
    script = f"""
import bpy

# CURATA SCENA
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

table_height = {table_height}
table_width = {table_width}
table_depth = {table_depth}
leg_count = {leg_count}
leg_thickness = {leg_thickness}
has_apron = {has_apron}
style_adjust = {style_adjust}

top_thickness = max(0.04, min(table_height * 0.3, table_height * 0.1 + style_adjust))
leg_height = table_height - top_thickness
leg_radius = leg_thickness / 2

# BLAT
bpy.ops.mesh.primitive_cube_add(size=2)
top = bpy.context.active_object
top.name = "TableTop"
top.scale = (table_width / 2, table_depth / 2, top_thickness / 2)
top.location = (0, 0, leg_height + top_thickness / 2)

# PICIOARE
x_offset = table_width / 2 - leg_radius
y_offset = table_depth / 2 - leg_radius

positions = [
    ( x_offset,  y_offset),
    (-x_offset,  y_offset),
    ( x_offset, -y_offset),
    (-x_offset, -y_offset),
][:leg_count]

legs = []
for i, (x, y) in enumerate(positions):
    bpy.ops.mesh.primitive_cylinder_add(radius=leg_radius, depth=leg_height)
    leg = bpy.context.active_object
    leg.name = f"Leg_{{i}}"
    leg.location = (x, y, leg_height / 2)
    legs.append(leg)

# SORT (cadru optional sub blat)
if has_apron == 1:
    apron_height = leg_thickness
    apron_z = leg_height - apron_height / 2
    apron_parts = []

    bpy.ops.mesh.primitive_cube_add(size=2)
    apron_front = bpy.context.active_object
    apron_front.scale = (
        table_width / 2 - leg_radius,
        apron_height / 2,
        apron_height / 2,
    )
    apron_front.location = (0, y_offset - leg_radius, apron_z)
    apron_parts.append(apron_front)

    bpy.ops.mesh.primitive_cube_add(size=2)
    apron_back = bpy.context.active_object
    apron_back.scale = (
        table_width / 2 - leg_radius,
        apron_height / 2,
        apron_height / 2,
    )
    apron_back.location = (0, -y_offset + leg_radius, apron_z)
    apron_parts.append(apron_back)

    bpy.ops.mesh.primitive_cube_add(size=2)
    apron_left = bpy.context.active_object
    apron_left.scale = (
        apron_height / 2,
        table_depth / 2 - leg_radius,
        apron_height / 2,
    )
    apron_left.location = (-x_offset + leg_radius, 0, apron_z)
    apron_parts.append(apron_left)

    bpy.ops.mesh.primitive_cube_add(size=2)
    apron_right = bpy.context.active_object
    apron_right.scale = (
        apron_height / 2,
        table_depth / 2 - leg_radius,
        apron_height / 2,
    )
    apron_right.location = (x_offset - leg_radius, 0, apron_z)
    apron_parts.append(apron_right)

# JOIN (uneste toate piesele intr-un singur obiect)
bpy.ops.object.select_all(action='DESELECT')
top.select_set(True)
for leg in legs:
    leg.select_set(True)
if has_apron == 1:
    for apron_part in apron_parts:
        apron_part.select_set(True)

bpy.context.view_layer.objects.active = top
bpy.ops.object.join()
"""

    return textwrap.dedent(script).strip()
