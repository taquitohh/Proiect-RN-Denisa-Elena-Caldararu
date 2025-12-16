import json
import re
from typing import Dict, Any, List, Tuple

class BlenderScriptGenerator:
    def __init__(self):
        self.default_params = {
            "size": 2.0,
            "radius": 1.0,
            "depth": 2.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "angle": 45.0,
            "scale": 1.5,
            "segments": 2
        }
        
        # Template-uri pentru toate intențiile suportate
        self.templates = {
            # Creare obiecte primitive
            "create_cube": "bpy.ops.mesh.primitive_cube_add(size={size}, location=({x}, {y}, {z}))",
            "create_cube_basic": "bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))",
            "create_cube_large": "bpy.ops.mesh.primitive_cube_add(size=4, location=(0, 0, 0))",
            "create_sphere": "bpy.ops.mesh.primitive_uv_sphere_add(radius={radius}, location=({x}, {y}, {z}))",
            "create_sphere_basic": "bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))",
            "create_sphere_large": "bpy.ops.mesh.primitive_uv_sphere_add(radius=3, location=(0, 0, 0))",
            "create_cylinder": "bpy.ops.mesh.primitive_cylinder_add(radius={radius}, depth={depth}, location=({x}, {y}, {z}))",
            "create_cylinder_basic": "bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, location=(0, 0, 0))",
            "create_cylinder_pillar": "bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=5, location=(0, 0, 0))",
            "create_cone": "bpy.ops.mesh.primitive_cone_add(radius1={radius}, depth={depth}, location=({x}, {y}, {z}))",
            "create_cone_basic": "bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, location=(0, 0, 0))",
            "create_cone_tall": "bpy.ops.mesh.primitive_cone_add(radius1=1, depth=4, location=(0, 0, 0))",
            "create_torus": "bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.25, location=({x}, {y}, {z}))",
            "create_torus_basic": "bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.25, location=(0, 0, 0))",
            "create_torus_ring": "bpy.ops.mesh.primitive_torus_add(major_radius=2, minor_radius=0.1, location=(0, 0, 0))",
            "create_plane": "bpy.ops.mesh.primitive_plane_add(size={size}, location=({x}, {y}, {z}))",
            "create_plane_floor": "bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))",
            "create_grid": "bpy.ops.mesh.primitive_grid_add(x_subdivisions=10, y_subdivisions=10, size=2, location=(0, 0, 0))",
            "create_monkey": "bpy.ops.mesh.primitive_monkey_add(size=2, location=(0, 0, 0))",
            "create_icosphere": "bpy.ops.mesh.primitive_ico_sphere_add(radius=1, subdivisions=2, location=(0, 0, 0))",
            "create_circle": "bpy.ops.mesh.primitive_circle_add(radius=1, location=(0, 0, 0))",
            "create_text": "bpy.ops.object.text_add(location=(0, 0, 0))",
            "create_pyramid": "bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=1, depth=2, location=(0, 0, 0))",
            
            # Transformări
            "move_object": "bpy.context.active_object.location = ({x}, {y}, {z})",
            "rotate_object": "import math\\nbpy.context.active_object.rotation_euler[2] = math.radians({angle})",
            "scale_object": "bpy.context.active_object.scale = ({scale}, {scale}, {scale})",
            "duplicate_object": "bpy.ops.object.duplicate_move()",
            
            # Ștergere
            "delete_all": "bpy.ops.object.select_all(action='SELECT')\\nbpy.ops.object.delete()",
            "delete_object": "bpy.ops.object.delete(use_global=False)",
            "delete_selected": "bpy.ops.object.delete(use_global=False)",
            
            # Modifiers
            "add_modifier_subsurf": "bpy.ops.object.modifier_add(type='SUBSURF')\\nbpy.context.object.modifiers['Subdivision'].levels = {segments}",
            "add_modifier_bevel": "bpy.ops.object.modifier_add(type='BEVEL')\\nbpy.context.object.modifiers['Bevel'].width = 0.1",
            "add_modifier_mirror": "bpy.ops.object.modifier_add(type='MIRROR')",
            "add_modifier_array": "bpy.ops.object.modifier_add(type='ARRAY')\\nbpy.context.object.modifiers['Array'].count = 5",
            "add_modifier_solidify": "bpy.ops.object.modifier_add(type='SOLIDIFY')\\nbpy.context.object.modifiers['Solidify'].thickness = 0.1",
            "add_modifier_boolean": "bpy.ops.object.modifier_add(type='BOOLEAN')",
            "add_modifier_decimate": "bpy.ops.object.modifier_add(type='DECIMATE')",
            "add_modifier_remesh": "bpy.ops.object.modifier_add(type='REMESH')",
            "add_modifier_smooth": "bpy.ops.object.modifier_add(type='SMOOTH')",
            "add_modifier_wave": "bpy.ops.object.modifier_add(type='WAVE')",
            "add_modifier_displace": "bpy.ops.object.modifier_add(type='DISPLACE')",
            "add_modifier_shrinkwrap": "bpy.ops.object.modifier_add(type='SHRINKWRAP')",
            "add_modifier_skin": "bpy.ops.object.modifier_add(type='SKIN')",
            "add_modifier_wireframe": "bpy.ops.object.modifier_add(type='WIREFRAME')",
            "add_modifier_cloth": "bpy.ops.object.modifier_add(type='CLOTH')",
            "add_modifier_ocean": "bpy.ops.object.modifier_add(type='OCEAN')",
            "add_modifier_screw": "bpy.ops.object.modifier_add(type='SCREW')",
            
            # Materiale
            "apply_material": "mat = bpy.data.materials.new(name='Material')\\nbpy.context.active_object.data.materials.append(mat)",
            "apply_material_red": "mat = bpy.data.materials.new(name='Red')\\nmat.diffuse_color = (1, 0, 0, 1)\\nbpy.context.active_object.data.materials.append(mat)",
            "apply_material_green": "mat = bpy.data.materials.new(name='Green')\\nmat.diffuse_color = (0, 1, 0, 1)\\nbpy.context.active_object.data.materials.append(mat)",
            "apply_material_blue": "mat = bpy.data.materials.new(name='Blue')\\nmat.diffuse_color = (0, 0, 1, 1)\\nbpy.context.active_object.data.materials.append(mat)",
            "apply_material_metal": "mat = bpy.data.materials.new(name='Metal')\\nmat.metallic = 1.0\\nbpy.context.active_object.data.materials.append(mat)",
            "apply_material_glass": "mat = bpy.data.materials.new(name='Glass')\\nmat.use_nodes = True\\nbpy.context.active_object.data.materials.append(mat)",
            
            # Lumini și cameră
            "create_light": "bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))",
            "create_light_point": "bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))",
            "create_light_sun": "bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))",
            "create_light_spot": "bpy.ops.object.light_add(type='SPOT', location=(0, 0, 5))",
            "create_camera": "bpy.ops.object.camera_add(location=(7, -7, 5))",
            
            # Export
            "export_obj": "bpy.ops.export_scene.obj(filepath='//export.obj')",
            "export_fbx": "bpy.ops.export_scene.fbx(filepath='//export.fbx')",
            "export_stl": "bpy.ops.export_mesh.stl(filepath='//export.stl')",
            "save_file": "bpy.ops.wm.save_mainfile(filepath='//scene.blend')",
            "render_scene": "bpy.ops.render.render(write_still=True)",
            
            # Editare mesh
            "subdivide_mesh": "bpy.ops.object.mode_set(mode='EDIT')\\nbpy.ops.mesh.subdivide()\\nbpy.ops.object.mode_set(mode='OBJECT')",
            "smooth_mesh": "bpy.ops.object.shade_smooth()",
            "join_objects": "bpy.ops.object.join()",
            "separate_objects": "bpy.ops.mesh.separate(type='LOOSE')",
            
            # Scene
            "create_complete_scene": "# Scenă completă\\nbpy.ops.mesh.primitive_plane_add(size=10)\\nbpy.ops.mesh.primitive_cube_add(location=(0,0,1))\\nbpy.ops.object.light_add(type='SUN', location=(5,5,10))\\nbpy.ops.object.camera_add(location=(7,-7,5))"
        }

    def generate_script(self, intent: str, extracted_params: Dict[str, Any]) -> str:
        """
        Generează scriptul final combinând intentia (clasa prezisă de AI)
        cu parametrii extrași din text.
        """
        # 1. Selectare template
        template = self.templates.get(intent, "")
        if not template:
            return "# Eroare: Nu am putut identifica un template pentru intentia: " + intent

        # 2. Combinare parametri impliciți cu cei extrași
        final_params = self.default_params.copy()
        final_params.update(extracted_params)

        # 3. Completare template
        try:
            script = "import bpy\n\n" + template.format(**final_params)
            return script
        except KeyError as e:
            return f"# Eroare generare script: Parametrul {e} lipsește din template."

    def extract_parameters_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extrage numere și entități simple folosind Regex (Rule-based extraction).
        Aceasta funcționează în tandem cu Rețeaua Neuronală care detectează DOAR intentia.
        """
        params = {}
        text = text.lower()

        # Extragere dimensiuni simple (ex: "raza 5", "inaltime 2")
        radius_match = re.search(r'(?:raza|radius)\s*(\d+(?:\.\d+)?)', text)
        if radius_match:
            params['radius'] = float(radius_match.group(1))

        size_match = re.search(r'(?:marime|dimensiune|latura|size)\s*(\d+(?:\.\d+)?)', text)
        if size_match:
            params['size'] = float(size_match.group(1))
            
        depth_match = re.search(r'(?:inaltime|lungime|depth|height)\s*(\d+(?:\.\d+)?)', text)
        if depth_match:
            params['depth'] = float(depth_match.group(1))

        # Extragere coordonate (ex: "pozitia 10 20 5")
        pos_match = re.search(r'(?:pozitia|locatia|la)\s*(\d+)\s+(\d+)\s+(\d+)', text)
        if pos_match:
            params['x'] = float(pos_match.group(1))
            params['y'] = float(pos_match.group(2))
            params['z'] = float(pos_match.group(3))

        # Extragere unghi (ex: "45 grade", "roteste cu 90")
        angle_match = re.search(r'(?:cu\s+)?(\d+(?:\.\d+)?)\s*(?:grade|degrees|°)?', text)
        if angle_match and ('rotat' in text or 'rote' in text or 'grade' in text):
            params['angle'] = float(angle_match.group(1))

        # Extragere factor de scalare (ex: "scala la 2", "mareste de 1.5 ori")
        scale_match = re.search(r'(?:scala|factor|de)\s*(\d+(?:\.\d+)?)', text)
        if scale_match and ('scal' in text or 'mare' in text):
            params['scale'] = float(scale_match.group(1))

        return params
