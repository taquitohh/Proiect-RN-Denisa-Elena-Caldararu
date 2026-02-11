# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

from __future__ import annotations

import textwrap


def generate_stove_script(
    stove_height: float,
    stove_width: float,
    stove_depth: float,
    oven_height_ratio: float,
    handle_length: float,
    glass_thickness: float,
    style_variant: int,
) -> str:
    # Normalizeaza optiunile discrete la intervalele asteptate.
    style_variant = int(style_variant)

    # Construieste un script Blender cu pasi expliciti de geometrie.
    script = f"""
import bpy

# CURATA SCENA
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

stove_height = {stove_height}
stove_width = {stove_width}
stove_depth = {stove_depth}
oven_height_ratio = {oven_height_ratio}
handle_length = {handle_length}
glass_thickness = {glass_thickness}
style_variant = {style_variant}

# CORP
bpy.ops.mesh.primitive_cube_add(size=2)
body = bpy.context.active_object
body.name = "StoveBody"
body.scale = (stove_width / 2, stove_depth / 2, stove_height / 2)
body.location = (0, 0, stove_height / 2)

# CAVITATE CUPTOR (decupaj boolean)
oven_height = stove_height * oven_height_ratio
cavity_width = stove_width * 0.75
cavity_depth = stove_depth * 0.7
cavity_height = oven_height * 0.7
cavity_z = cavity_height / 2 + stove_height * 0.1
cavity_y = stove_depth / 2 - cavity_depth / 2 - 0.02

bpy.ops.mesh.primitive_cube_add(size=2)
cavity = bpy.context.active_object
cavity.scale = (cavity_width / 2, cavity_depth / 2, cavity_height / 2)
cavity.location = (0, cavity_y, cavity_z)

modifier = body.modifiers.new(name="Cavity", type='BOOLEAN')
modifier.operation = 'DIFFERENCE'
modifier.object = cavity

bpy.context.view_layer.objects.active = body
bpy.ops.object.modifier_apply(modifier="Cavity")

bpy.ops.object.select_all(action='DESELECT')
cavity.select_set(True)
bpy.ops.object.delete(use_global=False)

# PLITA
cooktop_thickness = max(0.02, stove_height * 0.04)
bpy.ops.mesh.primitive_cube_add(size=2)
cooktop = bpy.context.active_object
cooktop.scale = (stove_width / 2, stove_depth / 2, cooktop_thickness / 2)
cooktop.location = (0, 0, stove_height + cooktop_thickness / 2)

# ARZATOARE (4)
burners = []
burner_radius = stove_width * 0.08
burner_height = cooktop_thickness * 0.6
x_offset = stove_width * 0.2
y_offset = stove_depth * 0.2

for x in (-x_offset, x_offset):
    for y in (-y_offset, y_offset):
        bpy.ops.mesh.primitive_cylinder_add(radius=burner_radius, depth=burner_height)
        burner = bpy.context.active_object
        burner.location = (x, y, stove_height + cooktop_thickness + burner_height / 2)
        burners.append(burner)

# BUTOANE (6)
knobs = []
knob_radius = max(0.01, stove_width * 0.02)
knob_depth = max(0.02, stove_depth * 0.06)
knob_y = stove_depth / 2 + knob_depth / 2
knob_z = stove_height * 0.75
knob_x_offsets = [
    -stove_width * 0.25,
    -stove_width * 0.1,
    stove_width * 0.05,
    stove_width * 0.2,
    -stove_width * 0.35,
    stove_width * 0.35,
]

for x in knob_x_offsets:
    bpy.ops.mesh.primitive_cylinder_add(radius=knob_radius, depth=knob_depth)
    knob = bpy.context.active_object
    knob.rotation_euler = (1.5708, 0.0, 0.0)
    knob.location = (x, knob_y, knob_z)
    knobs.append(knob)

# USA CUPTOR + STICLA
front_y = stove_depth / 2
bpy.ops.mesh.primitive_cube_add(size=2)
door = bpy.context.active_object
door_thickness = max(0.02, stove_depth * 0.06)
door.scale = (stove_width * 0.42, door_thickness / 2, oven_height * 0.38)
door.location = (0, front_y + door_thickness / 2, oven_height * 0.5)

bpy.ops.mesh.primitive_cube_add(size=2)
window = bpy.context.active_object
window.scale = (stove_width * 0.28, glass_thickness / 2, oven_height * 0.22)
window.location = (0, front_y + door_thickness / 2 + glass_thickness / 2, oven_height * 0.5)

# MANER USA
bpy.ops.mesh.primitive_cylinder_add(radius=knob_radius, depth=handle_length)
handle = bpy.context.active_object
handle.rotation_euler = (0.0, 1.5708, 0.0)
handle.location = (0, front_y + door_thickness + knob_radius, oven_height * 0.78)

# JOIN (uneste toate piesele intr-un singur obiect)
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
cooktop.select_set(True)
door.select_set(True)
window.select_set(True)
handle.select_set(True)
for burner in burners:
    burner.select_set(True)
for knob in knobs:
    knob.select_set(True)

bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
"""

    return textwrap.dedent(script).strip()
