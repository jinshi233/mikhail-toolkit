"""Close-up render of one part, for checking small details like the visor."""
import sys
import bpy
import math
from mathutils import Vector
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from mikhail import config as _cfg
except ImportError:
    import config as _cfg

args = sys.argv[sys.argv.index("--") + 1:]
fbx, out_png, part = args[0], args[1], args[2]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)

target = None
for o in bpy.data.objects:
    if o.type == "MESH" and part in o.name:
        target = o
        break
if target is None:
    raise SystemExit(f"no object matching {part}")

lo = Vector((1e18,) * 3)
hi = Vector((-1e18,) * 3)
for c in target.bound_box:
    w = target.matrix_world @ Vector(c)
    for i in range(3):
        lo[i] = min(lo[i], w[i])
        hi[i] = max(hi[i], w[i])
centre = (lo + hi) / 2
size = max((hi - lo)[:])

scn = bpy.context.scene
scn.render.engine = "BLENDER_EEVEE_NEXT"
scn.render.resolution_x = 900
scn.render.resolution_y = 900
scn.world = bpy.data.worlds.new("W")
scn.world.use_nodes = True
scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.38, 1)
scn.world.node_tree.nodes["Background"].inputs[1].default_value = 1.5

for rot, e in (((0.9, 0.2, 0.7), 6.0), ((1.2, -0.4, 2.5), 3.0)):
    ld = bpy.data.lights.new("L", "SUN")
    ld.energy = e
    ob = bpy.data.objects.new("L", ld)
    ob.rotation_euler = rot
    scn.collection.objects.link(ob)

cam_d = bpy.data.cameras.new("C")
cam_d.lens = 85
cam = bpy.data.objects.new("C", cam_d)
scn.collection.objects.link(cam)
scn.camera = cam
# three-quarter front, slightly above
cam.location = centre + Vector((size * 0.9, -size * 1.7, size * 0.25))
d = centre - cam.location
cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

out = out_png if out_png.endswith(".png") else out_png + ".png"
scn.render.filepath = out
scn.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("rendered", out)
