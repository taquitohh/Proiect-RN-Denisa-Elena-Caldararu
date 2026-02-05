"""Deterministic Blender script generator for parametric chairs."""

from __future__ import annotations

import textwrap


def generate_chair_script(
    seat_height: float,
    seat_width: float,
    seat_depth: float,
    leg_count: int,
    leg_thickness: float,
    has_backrest: int,
    backrest_height: float,
    style_variant: int,
) -> str:
    """Return a Blender (bpy) script that builds a simple parametric chair."""
    # Clamp safety to keep geometry sane in Blender while staying deterministic.
    leg_count = max(3, min(5, int(leg_count)))
    has_backrest = 1 if int(has_backrest) == 1 else 0
    backrest_height = max(0.0, float(backrest_height))

    # Style variant can tweak a subtle offset or scale while remaining deterministic.
    style_offset = 0.0 if int(style_variant) == 0 else (0.02 if int(style_variant) == 1 else -0.02)

    script = f"""
import bpy

# Parameters from the UI
seat_height = {seat_height}
seat_width = {seat_width}
seat_depth = {seat_depth}
leg_count = {leg_count}
leg_thickness = {leg_thickness}
has_backrest = {has_backrest}
backrest_height = {backrest_height}
style_offset = {style_offset}

# Optional: clear the default scene for a clean result
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Seat
bpy.ops.mesh.primitive_cube_add(size=1)
seat = bpy.context.active_object
seat.name = "Seat"
seat.scale = (seat_width / 2.0, seat_depth / 2.0, seat_height / 2.0)
seat.location = (0.0, 0.0, seat_height / 2.0)

# Legs
leg_radius = leg_thickness / 2.0
leg_height = seat_height
x_offset = (seat_width / 2.0) - leg_radius + style_offset
y_offset = (seat_depth / 2.0) - leg_radius + style_offset

leg_positions = [
    ( x_offset,  y_offset),
    (-x_offset,  y_offset),
    ( x_offset, -y_offset),
    (-x_offset, -y_offset),
]

# Trim or extend leg positions based on leg_count
leg_positions = leg_positions[:leg_count]

for idx, (x_pos, y_pos) in enumerate(leg_positions, start=1):
    bpy.ops.mesh.primitive_cylinder_add(radius=leg_radius, depth=leg_height)
    leg = bpy.context.active_object
    leg.name = f"Leg_{{idx}}"
    leg.location = (x_pos, y_pos, leg_height / 2.0)

# Backrest (optional)
if has_backrest == 1:
    bpy.ops.mesh.primitive_cube_add(size=1)
    backrest = bpy.context.active_object
    backrest.name = "Backrest"
    backrest.scale = (seat_width / 2.0, leg_thickness, backrest_height / 2.0)
    backrest.location = (0.0, -seat_depth / 2.0 + leg_thickness, seat_height + backrest_height / 2.0)
"""

    return textwrap.dedent(script).strip() + "\n"
