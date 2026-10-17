"""Verify an exported FBX: geometry, faces, scale, bones, and real deformation."""
import sys
import bpy
import mathutils

args = sys.argv[sys.argv.index("--") + 1:]
for path in args:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path)
    fname = path.replace("\\", "/").split("/")[-1]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    print(f"\n===== {fname}")
    if not meshes:
        print("  NO MESH")
        continue
    tv = tf = 0
    lo = [1e18] * 3
    hi = [-1e18] * 3
    for o in meshes:
        me = o.data
        me.calc_loop_triangles()
        tv += len(me.vertices)
        tf += len(me.loop_triangles)
        for c in me.vertices:
            w = o.matrix_world @ c.co
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])
    span = [hi[i] - lo[i] for i in range(3)]
    uvs = sorted({l.name for o in meshes for l in o.data.uv_layers})
    print(f"  meshes {len(meshes)}  vertices {tv:,}  faces {tf:,}")
    print(f"  span  X{span[0]:.2f} Y{span[1]:.2f} Z{span[2]:.2f} "
          f"(TSF ~15-21 m tall)")
    print(f"  uv layers {uvs}")
    if not arms:
        continue
    arm = arms[0]
    print(f"  bones {len(arm.data.bones)}")

    for target in ("Head_M", "Elbow_L", "Knee_L"):
        b = arm.data.bones.get(target)
        if not b:
            print(f"  {target}: MISSING")
            continue
        piv = (arm.matrix_world @ b.head_local).copy()
        before = []
        for o in meshes:
            g = o.vertex_groups.get(target)
            if not g:
                continue
            gi = g.index
            idx = [v.index for v in o.data.vertices
                   if any(ge.group == gi and ge.weight > 0.5 for ge in v.groups)]
            if not idx:
                continue
            before.append((o, idx,
                           [(o.matrix_world @ o.data.vertices[i].co).copy()
                            for i in idx]))
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        arm.data.edit_bones[target].transform(
            mathutils.Matrix.Rotation(0.6, 4, "X"))
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        moved = checked = 0
        for o, idx, old in before:
            oe = o.evaluated_get(dg)
            mev = oe.to_mesh()
            for k, i in enumerate(idx):
                checked += 1
                if (oe.matrix_world @ mev.vertices[i].co - old[k]).length > 1e-4:
                    moved += 1
            oe.to_mesh_clear()
        print(f"  rotate {target}: {moved}/{checked} weighted verts moved "
              f"(pivot Y={piv[1]:.2f})")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=path)
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        arm = arms[0]
