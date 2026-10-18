#!/usr/bin/env python3
"""
blender_rig.py — run inside Blender to build a rigged model from the JSON
produced by export_rigged.py, then write FBX and glTF.

    blender --background --factory-startup --python blender_rig.py -- <json> <outdir>

Construction
------------
The game meshes are consecutively sliced out of a single vertex buffer and each
part mesh carries its own skeleton (the same joint tree, with per-part bind
poses).  We therefore:

  1. build ONE armature from the LOD-0 skeleton, reusing bone names (which are
     unique), so all part meshes can share it;
  2. create one Blender object per part mesh, all bound to that armature;
  3. assign weights with vertex groups named after the joints.

Because the source bind poses are all in the same world space, the armature
rest pose is simply the joint world transforms with the same axis/unit change
the vertices received, and the skin binding is exact by construction — no
heuristic joint placement needed.

Coordinate change: game is Y-up centimetres; we emit Blender Z-up metres.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

UNIT = 0.01
# Y-up (x, y, z) -> Z-up (x, -z, y)
AXIS = np.array([[1, 0, 0, 0],
                 [0, 0, -1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=np.float64)

# mesh part name -> texture-set stem
PART_ALIAS = {
    "l_b_mount": "mountb", "r_b_mount": "mountb",
    "l_g_mount": "mountg", "r_g_mount": "mountg",
}


def argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def to_blender_matrix(m16):
    """4x4 row-major game matrix -> Blender Matrix, with unit/axis change."""
    M = np.asarray(m16, dtype=np.float64).reshape(4, 4)
    return Matrix((AXIS @ M @ AXIS.T) * UNIT)


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


# ------------------------------------------------------------------ material
def make_material(name, textures, base_dir):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (350, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    have = {}
    for t in textures:
        p = os.path.join(base_dir, t["png"])
        if os.path.exists(p):
            have.setdefault(t["role"], p)
    if textures and not have:
        print(f"    ! material {name}: none of {len(textures)} texture files "
              f"found under {base_dir}")

    def img(path, non_color, loc):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.location = loc
        n.image = bpy.data.images.load(path)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        return n

    x = -450
    if "baseColor" in have:
        n = img(have["baseColor"], False, (x, 320))
        nt.links.new(n.outputs["Color"], bsdf.inputs["Base Color"])
    if "normal" in have:
        n = img(have["normal"], True, (x, 40))
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-120, 40)
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if "roughness" in have:
        n = img(have["roughness"], True, (x, -240))
        nt.links.new(n.outputs["Color"], bsdf.inputs["Roughness"])
    if "metalness" in have:
        n = img(have["metalness"], True, (x, -500))
        nt.links.new(n.outputs["Color"], bsdf.inputs["Metallic"])
    if "emissive" in have:
        # This is the baked `_em.png` (palette colours masked onto the model),
        # NOT the raw `_cm` region mask.  The mask is flat red/green/blue blobs
        # that mean "which UserColor slot", so using it as a colour painted whole
        # models with blotches and made the eye strip blue.
        # keep the surface opaque; FBX round-trips like to wire base-colour
        # alpha into this socket, which made every mesh render as a black
        # silhouette
        bsdf.inputs["Alpha"].default_value = 1.0
        n = img(have["emissive"], False, (x, -760))
        nt.links.new(n.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.4
    return mat


# ------------------------------------------------------------------- armature
def build_armature(nodes, skel, name, all_skeletons):
    """
    Build a clean, animation-ready FK armature.

    The file's scene graph is a Maya-style rig where every animated joint `X`
    hangs off a static `X_offset` helper node.  Emitting those helpers produces
    a forest of extra bones that clutter the viewport and make the joints look
    scattered, and the joints do not form direct FK chains.

    We therefore keep ONLY the joints the skin actually references, and make
    them direct parent/child: each joint's parent becomes the nearest ancestor
    that is itself a referenced joint.  Because every bone's rest matrix is set
    to that joint's world transform, the bind is exact either way, so this
    changes only the presentation and the animation handle, not the skinning.
    """
    by_uid = {n["uid"]: n for n in nodes}
    by_name = {n["name"]: n for n in nodes}

    # 1. every joint referenced by any part mesh
    joints = {}
    for s in all_skeletons:
        for i, uid in enumerate(s["nodes"]):
            nm = s["nodeNames"][i] if i < len(s["nodeNames"]) else None
            if not nm:
                continue
            n = by_uid.get(uid) or by_name.get(nm)
            if n is not None:
                joints[nm] = n

    # 2. walk up the scene graph to the nearest ancestor that is a joint
    def nearest_joint_parent(n):
        p = n.get("parentIndex")
        hops = 0
        while p is not None and 0 <= p < len(nodes) and hops < 64:
            pn = nodes[p]["name"]
            if pn in joints:
                return pn
            p = nodes[p].get("parentIndex")
            hops += 1
        return None

    parent_name = {}
    for nm, n in joints.items():
        pn = nearest_joint_parent(n)
        if pn and pn != nm:
            parent_name[nm] = pn

    arm_data = bpy.data.armatures.new(name + "_armature")
    arm = bpy.data.objects.new(name + "_armature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm

    # world (rest) matrices in Blender space
    W = {nm: to_blender_matrix(n["worldMatrix"]) for nm, n in joints.items()}

    # create parents before children
    order, seen = [], set()

    def visit(nm):
        if nm in seen:
            return
        seen.add(nm)
        pn = parent_name.get(nm)
        if pn:
            visit(pn)
        order.append(nm)

    for nm in joints:
        visit(nm)

    bpy.ops.object.mode_set(mode="EDIT")
    ebs = arm_data.edit_bones
    created = {}
    for nm in order:
        created[nm] = ebs.new(nm)

    children = {}
    for nm, pn in parent_name.items():
        children.setdefault(pn, []).append(nm)

    for nm, eb in created.items():
        wm = W[nm]
        head = wm.translation
        kids = sorted(children.get(nm, []))
        if kids:
            # point the bone at its first child so the chain reads naturally
            tail = W[kids[0]].translation
        else:
            d = wm.to_3x3() @ Vector((0.0, 1.0, 0.0))
            if d.length < 1e-9:
                d = Vector((0.0, 0.0, 1.0))
            tail = head + d.normalized() * 0.06
        if (tail - head).length < 1e-6:
            tail = head + Vector((0.0, 0.0, 0.03))
        eb.head = head
        eb.tail = tail
        eb.use_connect = False

    for nm, eb in created.items():
        pn = parent_name.get(nm)
        if pn and pn in created:
            eb.parent = created[pn]
            eb.use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    nparent = sum(1 for nm in created if parent_name.get(nm) in created)
    print(f"    armature: {len(created)} joints, {nparent} FK links "
          f"(offset helpers folded into parents)")
    return arm, created


# ----------------------------------------------------------------------- mesh
def build_part(data, mesh, arm, base_dir, name, mat_cache):
    vn = mesh["vertexCount"]
    pos = np.asarray(mesh["positions"], dtype=np.float64)
    nrm = np.asarray(mesh["normals"], dtype=np.float64)
    uv0 = np.asarray(mesh["uv0"], dtype=np.float64)

    # game Y-up -> Blender Z-up, metres already applied on export
    p = np.empty_like(pos)
    p[:, 0] = pos[:, 0]
    p[:, 1] = -pos[:, 2]
    p[:, 2] = pos[:, 1]
    n = np.empty_like(nrm)
    n[:, 0] = nrm[:, 0]
    n[:, 1] = -nrm[:, 2]
    n[:, 2] = nrm[:, 1]

    # Name the object after the BODY PART, not its file position: the mesh order
    # varies per unit, so "p3" tells the user nothing and hid a real bug (the
    # F-15 family lists both arms last, Type-94 lists `chest` last).
    part = mesh.get("part") or f"p{mesh['index']}"
    me = bpy.data.meshes.new(f"{name}_{part}")

    # Faces: <unit>.indices.npy holds one contiguous run per mesh, in mesh
    # order.  Per-mesh submeshes (F-14 / MiG-21 / YF-23 Pav-1) renumber their
    # indices from 0, so the run starts at `indexStart`; a unit with one shared
    # global buffer rebases by `vertexStart` instead.
    idx = data["_indices"]
    v0 = mesh["vertexStart"]
    vn = mesh["vertexCount"]
    local = mesh.get("localIndices", False)
    # Global-index units keep ABSOLUTE vertex numbers, so a mesh's run must be
    # rebased by its own vertex base before it can index this slice -- without
    # that, every index outside [0, vn) fails the bound check below and the
    # triangles are silently dropped (loose vertices, exploded limbs).
    # Per-mesh submeshes are already renumbered from 0.
    i0 = mesh.get("indexStart", 0) if local else v0
    shift = v0 if not local else 0
    ntri = sum((db.get("indexCount") or 0) // 3 for db in mesh["drawblocks"])
    polys = []
    seg = idx[i0:i0 + ntri * 3] if idx else []
    for k in range(0, len(seg) - 2, 3):
        a, b, c = seg[k] - shift, seg[k + 1] - shift, seg[k + 2] - shift
        if 0 <= a < vn and 0 <= b < vn and 0 <= c < vn \
                and a != b and b != c and a != c:
            polys.append((a, b, c))
    me.from_pydata([tuple(v) for v in p], [], polys)
    me.validate(verbose=False)
    me.update()

    # UV: layers are indexed per *loop*, not per vertex.
    # The scene stores per-texture UV channel indices; the game's basecolor /
    # normal / ORM maps all sample channel 0, so export exactly one layer and
    # make it the active render layer.  Shipping a second layer caused the FBX
    # to sample the wrong channel, which scrambled the textures.
    me.update()
    lv = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", lv)
    arr = np.asarray(mesh["uv0"], dtype=np.float64)
    layer = me.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", arr[lv].ravel().tolist())
    layer.active = True
    layer.active_render = True

    # normals
    me.update()
    me.normals_split_custom_set_from_vertices([tuple(v) for v in n])

    obj = bpy.data.objects.new(me.name, me)
    bpy.context.collection.objects.link(obj)

    # vertex groups / weights (vectorised: 4 influence slots)
    names = mesh["jointNames"]
    weights = np.asarray(mesh["weights"], dtype=np.float64)
    joints = np.asarray(mesh["joints"], dtype=np.int64)
    nj = len(names)

    # joint index -> group, created lazily
    groups = {}
    used = np.unique(joints[(weights > 1e-6) & (joints >= 0) & (joints < nj)]) \
        if nj else np.zeros(0, np.int64)
    for jt in used.tolist():
        nm = names[jt] if 0 <= jt < nj else None
        if not nm:
            continue
        g = obj.vertex_groups.get(nm) or obj.vertex_groups.new(name=nm)
        groups[int(jt)] = g

    for slot in range(4):
        wcol = weights[:, slot]
        jcol = joints[:, slot]
        nz = np.nonzero((wcol > 1e-6) & (jcol >= 0) & (jcol < nj))[0]
        if not len(nz):
            continue
        # bucket by (joint, quantised weight) and issue one add() per bucket
        q = np.round(wcol[nz], 4)
        for jt in np.unique(jcol[nz]).tolist():
            g = groups.get(int(jt))
            if g is None:
                continue
            sel = nz[jcol[nz] == jt]
            wq = q[jcol[nz] == jt]
            for wv in np.unique(wq).tolist():
                ids = sel[wq == wv].tolist()
                if ids:
                    g.add(ids, float(wv), "REPLACE")

    # armature binding (skipped for unrigged static props)
    if arm is not None:
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
        obj.parent = arm

    # material: each part mesh maps to its own part's texture set
    part = mesh.get("part")
    # Materials are keyed by TEXTURE SET, which is not always the part label:
    # l_b_mount and r_b_mount both use the "mountb" set, and j_unit uses "unitj".
    texset = mesh.get("texset") or part
    mats = [m for m in data["materials"] if m["part"] == texset] if texset else []
    if not mats:
        # Never fall back to "the first set in the file": that builds a material
        # from the WRONG part's maps and caches it under that part's name, so the
        # real mesh of that part later receives the crippled material and loses
        # its normal / roughness / metallic maps.  Use any set carrying a base
        # colour and key it under the missing part instead.
        mats = [m for m in data["materials"] if m["role"] == "baseColor"][:1]
        print(f"    ! mesh {mesh['index']}: no texture set for part {part!r}; "
              f"using {mats[0]['part'] if mats else None!r} base colour only")
    if mats:
        key = texset if texset else mats[0]["part"]
        if key not in mat_cache:
            mat_cache[key] = make_material(f"{name}_{key}", mats, base_dir)
        obj.data.materials.append(mat_cache[key])
    return obj


def main():
    a = argv()
    if len(a) < 2:
        print("usage: -- <json> <outdir>")
        return
    js, outdir = a[0], a[1]
    os.makedirs(outdir, exist_ok=True)
    data = json.load(open(js))
    name = data["name"]
    # textures live in <unit>/textures/ next to the json, and the json's "png"
    # values are relative to the unit directory
    base_dir = os.path.dirname(os.path.abspath(js))
    print(f"== {name}: {len(data['meshes'])} part meshes, {len(data['nodes'])} nodes")

    # index buffer is shared; it was written alongside the json
    idx_path = os.path.join(os.path.dirname(js), name + ".indices.npy")
    data["_indices"] = np.load(idx_path).tolist() if os.path.exists(idx_path) else []

    clear()
    # `kakasi` is a static prop with no skeletons at all; build it unrigged
    # rather than failing, and let the caller export a plain mesh
    if data["skeletons"]:
        arm, bones = build_armature(data["nodes"], data["skeletons"][0], name,
                                    data["skeletons"])
    else:
        arm, bones = None, {}
        print("    no skeletons: exporting a static mesh")
    mat_cache = {}
    objs = []
    for mesh in data["meshes"]:
        if mesh.get("lod") not in (None, 0):
            continue
        o = build_part(data, mesh, arm, base_dir, name, mat_cache)
        objs.append(o)
    print(f"    {len(objs)} objects in LOD0")

    # strict whitelist: only the armature and the part meshes we built may be
    # exported.  A stray 42-vertex "Icosphere" has been observed appearing in
    # the glTF stage, so drop anything we did not create.
    allowed = {o.name for o in objs} | ({arm.name} if arm else set())
    for o in list(bpy.data.objects):
        if o.name not in allowed:
            print(f"    removing stray object {o.name!r} ({o.type})")
            bpy.data.objects.remove(o, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    if arm:
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm

    fbx = os.path.join(outdir, name + ".fbx")
    # The game's UVs agree with glTF but come out V-flipped through Blender's
    # FBX path, so flip V for the FBX only (V' = 1 - V) and leave the glTF
    # untouched.  Toggled by --no-flip-v.
    flip_v = "--no-flip-v" not in sys.argv
    if flip_v:
        for o in objs:
            for layer in o.data.uv_layers:
                n = len(layer.data)
                buf = np.empty(n * 2, dtype=np.float64)
                layer.data.foreach_get("uv", buf)
                buf = buf.reshape(-1, 2)
                buf[:, 1] = 1.0 - buf[:, 1]
                layer.data.foreach_set("uv", buf.ravel().tolist())
                layer.data.update()
        print(f"    flipped V on {len(objs)} meshes (FBX convention)")

    bpy.ops.export_scene.fbx(
        filepath=fbx, use_selection=True, object_types={"MESH", "ARMATURE"},
        path_mode="COPY", embed_textures=True, mesh_smooth_type="FACE",
        use_mesh_modifiers=False, bake_space_transform=False,
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
    )
    print(f"    wrote {fbx} ({os.path.getsize(fbx)/1e6:.1f} MB)")

    # restore V before the glTF pass so both formats describe the same surface
    if flip_v:
        for o in objs:
            for layer in o.data.uv_layers:
                n = len(layer.data)
                buf = np.empty(n * 2, dtype=np.float64)
                layer.data.foreach_get("uv", buf)
                buf = buf.reshape(-1, 2)
                buf[:, 1] = 1.0 - buf[:, 1]
                layer.data.foreach_set("uv", buf.ravel().tolist())
                layer.data.update()

    glb = os.path.join(outdir, name + ".glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB",
                              use_selection=True, export_yup=True,
                              export_apply=False)
    print(f"    wrote {glb} ({os.path.getsize(glb)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
