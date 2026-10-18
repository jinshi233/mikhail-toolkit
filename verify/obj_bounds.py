"""Per-object bounds and vertex count in an exported FBX, plus bound bones."""
import sys
import bpy
from mathutils import Vector
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from mikhail import config as _cfg
except ImportError:
    import config as _cfg

fbx = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
arm = arms[0] if arms else None

rows = []
for o in meshes:
    lo = Vector((1e18,) * 3)
    hi = Vector((-1e18,) * 3)
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        for i in range(3):
            lo[i] = min(lo[i], w[i])
            hi[i] = max(hi[i], w[i])
    groups = [g.name for g in o.vertex_groups]
    rows.append((o.name, len(o.data.vertices), lo, hi, hi - lo, groups))

allz = [r[2].z for r in rows] + [r[3].z for r in rows]
H = max(allz) - min(allz)
print(f"model height (Z) = {H:.2f} m, objects={len(rows)}")
print(f"{'object':<24}{'verts':>9}{'size(m) X,Y,Z':>24}{'coverZ':>9}  bones")
for name, nv, lo, hi, sz, groups in rows:
    cover = (hi.z - lo.z) / H if H else 0
    print(f"{name:<24}{nv:>9,}   {sz.x:>6.2f} {sz.y:>6.2f} {sz.z:>6.2f}"
          f"{cover:>8.0%}  {len(groups)}")
    print(f"    z {lo.z:>7.2f}..{hi.z:<7.2f}  groups: {groups[:8]}")
