from __future__ import annotations

import textwrap


def generate_fridge_script(
    fridge_height: float,
    fridge_width: float,
    fridge_depth: float,
    door_thickness: float,
    handle_length: float,
    freezer_ratio: float,
    freezer_position: int,
    style_variant: int,
) -> str:
    # Normalizeaza optiunile discrete la intervalele asteptate.
    freezer_position = 1 if int(freezer_position) == 1 else 0
    style_variant = int(style_variant)

    # Construieste un script Blender cu pasi expliciti de geometrie.
    script = f"""
import bpy

# CURATA SCENA
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

fridge_height = {fridge_height}
fridge_width = {fridge_width}
fridge_depth = {fridge_depth}
door_thickness = {door_thickness}
handle_length = {handle_length}
freezer_ratio = {freezer_ratio}
freezer_position = {freezer_position}
style_variant = {style_variant}

# CARCASA
bpy.ops.mesh.primitive_cube_add(size=2)
carcass = bpy.context.active_object
carcass.name = "FridgeCarcass"
carcass.scale = (fridge_width / 2, fridge_depth / 2, fridge_height / 2)
carcass.location = (0, 0, fridge_height / 2)

front_y = fridge_depth / 2
panel_gap = max(0.002, door_thickness * 0.2)

freezer_height = fridge_height * freezer_ratio
fridge_section_height = fridge_height - freezer_height

if freezer_position == 0:
    top_height = freezer_height
    bottom_height = fridge_section_height
else:
    top_height = fridge_section_height
    bottom_height = freezer_height

# PANOURI USI
bpy.ops.mesh.primitive_cube_add(size=2)
top_door = bpy.context.active_object
top_door.name = "TopDoor"
top_door.scale = (fridge_width / 2 - panel_gap, door_thickness / 2, top_height / 2 - panel_gap)
top_door.location = (0, front_y + door_thickness / 2, fridge_height - top_height / 2)

bpy.ops.mesh.primitive_cube_add(size=2)
bottom_door = bpy.context.active_object
bottom_door.name = "BottomDoor"
bottom_door.scale = (fridge_width / 2 - panel_gap, door_thickness / 2, bottom_height / 2 - panel_gap)
bottom_door.location = (0, front_y + door_thickness / 2, bottom_height / 2)

# MANERE
handles = []
handle_radius = max(0.01, door_thickness * 0.35)
handle_x = fridge_width / 2 - panel_gap - handle_radius
handle_y = front_y + door_thickness + handle_radius

bpy.ops.mesh.primitive_cylinder_add(radius=handle_radius, depth=handle_length)
handle_top = bpy.context.active_object
handle_top.name = "HandleTop"
handle_top.rotation_euler = (0.0, 0.0, 1.5708)
handle_top.location = (handle_x, handle_y, fridge_height - top_height * 0.5)
handles.append(handle_top)

bpy.ops.mesh.primitive_cylinder_add(radius=handle_radius, depth=handle_length)
handle_bottom = bpy.context.active_object
handle_bottom.name = "HandleBottom"
handle_bottom.rotation_euler = (0.0, 0.0, 1.5708)
handle_bottom.location = (handle_x, handle_y, bottom_height * 0.5)
handles.append(handle_bottom)

# JOIN (uneste toate piesele intr-un singur obiect)
bpy.ops.object.select_all(action='DESELECT')
carcass.select_set(True)
top_door.select_set(True)
bottom_door.select_set(True)
for handle in handles:
    handle.select_set(True)

bpy.context.view_layer.objects.active = carcass
bpy.ops.object.join()
"""

    return textwrap.dedent(script).strip()
