"""Acceptance: per unit report object split, bindings, loose vertices, deformation."""
import os
import sys
import glob
import bpy

roots = sys.argv[sys.argv.index("--") + 1:]
rows = []
for root in roots:
    for path in sorted(glob.glob(os.path.join(root, "*", "*.fbx"))):
        unit = os.path.basename(os.path.dirname(path))
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            bpy.ops.import_scene.fbx(filepath=path)
        except Exception as exc:
            print(f"{unit}: IMPORT FAIL {exc}")
            continue
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        arm = arms[0] if arms else None
        bn = set(arm.data.bones.keys()) if arm else set()

        tv = tf = loose = unbound = orphan = 0
        for o in meshes:
            me = o.data
            me.calc_loop_triangles()
            tv += len(me.vertices)
            tf += len(me.loop_triangles)
            used = set()
            for p in me.polygons:
                used.update(p.vertices)
            loose += len(me.vertices) - len(used)
            orphan += sum(1 for g in o.vertex_groups if g.name not in bn)
            for v in me.vertices:
                if sum(g.weight for g in v.groups
                       if o.vertex_groups[g.group].name in bn) <= 1e-6:
                    unbound += 1

        # deform every object's dominant bone
        ok = tot = 0
        if arm:
            for o in meshes:
                best = None
                for g in o.vertex_groups:
                    if g.name not in bn:
                        continue
                    gi = g.index
                    n = sum(1 for v in o.data.vertices
                            if any(ge.group == gi and ge.weight > 0.3
                                   for ge in v.groups))
                    if n and (best is None or n > best[0]):
                        best = (n, g.name)
                if not best:
                    continue
                tgt = best[1]
                g = o.vertex_groups[tgt]
                gi = g.index
                idx = [v.index for v in o.data.vertices
                       if any(ge.group == gi and ge.weight > 0.3
                              for ge in v.groups)]
                bpy.context.view_layer.objects.active = arm
                bpy.ops.object.mode_set(mode="POSE")
                pb = arm.pose.bones[tgt]
                pb.rotation_mode = "XYZ"
                pb.rotation_euler[0] += 0.5
                bpy.context.view_layer.update()
                dg = bpy.context.evaluated_depsgraph_get()
                oe = o.evaluated_get(dg)
                mev = oe.to_mesh()
                for i in idx:
                    tot += 1
                    if (oe.matrix_world @ mev.vertices[i].co
                            - o.matrix_world @ o.data.vertices[i].co).length > 1e-5:
                        ok += 1
                oe.to_mesh_clear()
                bpy.ops.object.mode_set(mode="OBJECT")
                pb.rotation_euler[0] -= 0.5
                bpy.context.view_layer.update()
        rows.append((unit, len(meshes), len(arms), tv, tf, loose, unbound,
                     orphan, ok, tot))

print("\n" + "=" * 104)
print(f"{'unit':<18}{'objs':>5}{'verts':>10}{'tris':>9}{'loose':>8}"
      f"{'unbound':>9}{'orphanVG':>9}{'deform':>13}")
print("-" * 104)
bad = []
for (u, no, na, tv, tf, lo, ub, og, ok, tt) in rows:
    fails = []
    if lo:
        fails.append("LOOSE")
    if ub:
        fails.append("UNBOUND")
    if og:
        fails.append("ORPHAN-VG")
    if tt and ok != tt:
        fails.append("DEFORM")
    if fails:
        bad.append((u, fails))
    d = f"{ok}/{tt}" if tt else "n/a"
    print(f"{u:<18}{no:>5}{tv:>10,}{tf:>9,}{lo:>8,}{ub:>9,}{og:>9}{d:>13}"
          + ("   <== " + ",".join(fails) if fails else ""))
print("-" * 104)
print(f"{len(rows)} units, {len(bad)} failing: {bad}")
