# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.

from __future__ import annotations

import textwrap


def generate_cabinet_script(
    cabinet_height: float,
    cabinet_width: float,
    cabinet_depth: float,
    wall_thickness: float,
    door_type: int,
    door_count: int,
    style_variant: int,
) -> str:
    # Normalizeaza optiunile discrete la intervalele asteptate.
    door_type = 1 if int(door_type) == 1 else 0
    door_count = 1 if int(door_count) == 1 else 2
    style_variant = int(style_variant)

    # Construieste un script Blender cu pasi expliciti de geometrie.
    script = f"""
import bpy

# CURATA SCENA
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

cabinet_height = {cabinet_height}
cabinet_width = {cabinet_width}
cabinet_depth = {cabinet_depth}
wall_thickness = {wall_thickness}
door_type = {door_type}
door_count = {door_count}
style_variant = {style_variant}

front_offset = cabinet_depth / 2
back_offset = -cabinet_depth / 2

# CARCASA
bpy.ops.mesh.primitive_cube_add(size=2)
carcass = bpy.context.active_object
carcass.name = "CabinetCarcass"
carcass.scale = (cabinet_width / 2, cabinet_depth / 2, cabinet_height / 2)
carcass.location = (0, 0, cabinet_height / 2)

# USI
inset_gap = wall_thickness * 0.4
frame_gap = wall_thickness * 0.2
door_gap = max(0.002, wall_thickness * 0.15)
base_door_thickness = max(0.015, wall_thickness * 0.8)
if style_variant == 1:
    base_door_thickness *= 1.1
elif style_variant == 2:
    base_door_thickness *= 0.9

if door_type == 0:
    door_depth = base_door_thickness
    door_center_y = front_offset + door_depth / 2
    door_width_total = cabinet_width - 2 * frame_gap
    door_height = cabinet_height - 2 * frame_gap
else:
    door_depth = base_door_thickness
    door_center_y = front_offset - inset_gap - door_depth / 2
    door_width_total = cabinet_width - 2 * (frame_gap + inset_gap)
    door_height = cabinet_height - 2 * (frame_gap + inset_gap)

door_width_total = max(0.1, door_width_total)
door_height = max(0.2, door_height)

if door_count == 1:
    bpy.ops.mesh.primitive_cube_add(size=2)
    door = bpy.context.active_object
    door.name = "Door_0"
    door.scale = (door_width_total / 2, door_depth / 2, door_height / 2)
    door.location = (0, door_center_y, cabinet_height / 2)
    doors = [door]
else:
    door_width_each = (door_width_total - frame_gap - door_gap) / 2
    x_offset = door_width_each / 2 + (frame_gap + door_gap) / 2
    doors = []
    for i, x in enumerate((-x_offset, x_offset)):
        bpy.ops.mesh.primitive_cube_add(size=2)
        door = bpy.context.active_object
        door.name = f"Door_{{i}}"
        door.scale = (door_width_each / 2, door_depth / 2, door_height / 2)
        door.location = (x, door_center_y, cabinet_height / 2)
        doors.append(door)

# MANERE
handles = []
handle_radius = max(0.005, wall_thickness * 0.25)
handle_length = max(0.03, wall_thickness * 1.5)
handle_y = door_center_y + door_depth / 2 + handle_length / 2
handle_z = cabinet_height * 0.55

if door_count == 1:
    handle_x = door_width_total / 2 - frame_gap - handle_radius
    bpy.ops.mesh.primitive_cylinder_add(radius=handle_radius, depth=handle_length)
    handle = bpy.context.active_object
    handle.name = "Handle_0"
    handle.rotation_euler = (0.0, 1.5708, 0.0)
    handle.location = (handle_x, handle_y, handle_z)
    handles.append(handle)
else:
    handle_x = door_gap / 2 + handle_radius
    for i, x in enumerate((-handle_x, handle_x)):
        bpy.ops.mesh.primitive_cylinder_add(radius=handle_radius, depth=handle_length)
        handle = bpy.context.active_object
        handle.name = f"Handle_{{i}}"
        handle.rotation_euler = (0.0, 1.5708, 0.0)
        handle.location = (x, handle_y, handle_z)
        handles.append(handle)

# JOIN (uneste toate piesele intr-un singur obiect)
bpy.ops.object.select_all(action='DESELECT')
carcass.select_set(True)
for door in doors:
    door.select_set(True)
for handle in handles:
    handle.select_set(True)

bpy.context.view_layer.objects.active = carcass
bpy.ops.object.join()
"""

    return textwrap.dedent(script).strip()
