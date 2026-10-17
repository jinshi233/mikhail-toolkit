"""Rebuild one unit from JSON and report, per mesh, which vertex groups exist,
which are missing from the armature, and how many vertices are actually bound."""
import sys
import os
import json
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
js = argv[0]
outdir = argv[1]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "brig", os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_rig.py"))
# do not execute blender_rig's main(); replicate its build steps instead
import numpy as np

data = json.load(open(js))
name = data["name"]
base_dir = os.path.dirname(os.path.abspath(js))
idx_path = os.path.join(os.path.dirname(js), name + ".indices.npy")
data["_indices"] = np.load(idx_path).tolist() if os.path.exists(idx_path) else []

rig = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().replace("\nmain()\n", "\n")
exec(compile(src, spec.origin, "exec"), rig.__dict__)

rig.clear()
arm, bones = rig.build_armature(data["nodes"], data["skeletons"][0], name,
                                data["skeletons"])
mat_cache = {}
print(f"\narmature bones: {len(bones)}")
print(f"armature has Hip_Unit5_R: {'Hip_Unit5_R' in bones}")
print(f"armature has Knee_L: {'Knee_L' in bones}")
print(f"armature has Hip_Unit2_R: {'Hip_Unit2_R' in bones}")

for mesh in data["meshes"]:
    if mesh.get("lod") not in (None, 0):
        continue
    o = rig.build_part(data, mesh, arm, base_dir, name, mat_cache)
    vg = [g.name for g in o.vertex_groups]
    missing = [g for g in vg if g not in bones]
    bound = sum(1 for v in o.data.vertices if len(v.groups) > 0)
    nj = len(mesh["jointNames"])
    jmax = max((j for j in mesh["joints"] if True), default=[])
    print(f"  mesh{mesh['index']:>3} lod={mesh.get('lod')} part={mesh.get('part'):<8} "
          f"verts={len(o.data.vertices):>6} jointNames={nj:>3} "
          f"vgroups={len(vg):>3} missing_from_arm={len(missing):>3} "
          f"bound={bound:>6} ({100*bound/max(1,len(o.data.vertices)):.1f}%)")
    if missing:
        print(f"        MISSING: {missing[:10]}")
