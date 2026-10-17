"""Render an exported unit with its textures so colours can be eyeballed."""
import os
import sys
import math
import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
fbx, out_png = args[0], args[1]
views = int(args[2]) if len(args) > 2 else 3

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
lo = Vector((1e18,) * 3)
hi = Vector((-1e18,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        for i in range(3):
            lo[i] = min(lo[i], w[i])
            hi[i] = max(hi[i], w[i])
centre = (lo + hi) / 2
size = max((hi - lo)[:])

scn = bpy.context.scene
scn.render.engine = "BLENDER_EEVEE_NEXT"
scn.render.resolution_x = 900
scn.render.resolution_y = 900
scn.render.film_transparent = False
scn.world = bpy.data.worlds.new("W")
scn.world.use_nodes = True
scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
scn.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

# sun + fill
for name, rot, e in (("key", (0.9, 0.2, 0.7), 4.0), ("fill", (1.2, -0.4, 2.5), 2.0)):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = e
    ob = bpy.data.objects.new(name, ld)
    ob.rotation_euler = rot
    scn.collection.objects.link(ob)

cam_d = bpy.data.cameras.new("C")
cam_d.lens = 60
cam = bpy.data.objects.new("C", cam_d)
scn.collection.objects.link(cam)
scn.camera = cam

for k in range(views):
    ang = math.radians(-35 + k * 90)
    dist = size * 1.9
    cam.location = centre + Vector((math.sin(ang) * dist, -math.cos(ang) * dist,
                                    size * 0.18))
    d = centre - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    out = os.path.abspath(out_png).replace(".png", f"_{k}.png")
    scn.render.filepath = out
    scn.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print("rendered", out, "exists:", os.path.exists(out))
