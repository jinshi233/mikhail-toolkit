#!/usr/bin/env python3
"""
Extract what the archive actually says about weapons.

The client ships no weapon name / faction table (no text hit for weapon names, no
config file).  What IS in the data:

  * every pose3d animation names the equipment it attaches, so the pose files
    give a real weapon <-> TSF association;
  * each weapon's own pose file lists its joint names, which describe function
    (barrel, magazine, blade ...).

Both are extracted here verbatim; nothing is inferred.
"""
import json
import os
import re
import sys
from collections import defaultdict

WS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "mikhail"))
from mikhail_assets import FpdArchive
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from mikhail import config as _cfg
except ImportError:
    import config as _cfg

GAME = _cfg.GAME
pack = FpdArchive(_cfg.PACK)

EQ = re.compile(rb'data/equip/([a-z]+)/([^/"]+)/model/([^/"]+)\.ma')
JOINT = re.compile(rb'<joint key="([^"]+)"')
POSE = re.compile(r'/pose3d/([^/]+)/([^/]+)\.xml$')

weapon_users = defaultdict(set)     # weapon id -> set of tsf ids
weapon_poses = defaultdict(set)     # weapon id -> pose file names (own folder)
tsf_weapons = defaultdict(set)      # tsf -> set of weapon ids
joints = {}
pose_files = 0

for e in pack.entries:
    if not e.name.endswith('.xml') or '/pose3d/' in e.name and e.size == 0:
        continue
    if '/pose3d/' not in e.name:
        continue
    m = POSE.search(e.name)
    if not m:
        continue
    owner, fname = m.group(1), m.group(2)
    try:
        d = pack.payload(e.name)
    except Exception:
        continue
    pose_files += 1
    found = {g[2].decode() for g in EQ.findall(d)}
    for w in found:
        if owner == 'weapon':
            weapon_poses[w].add(fname)
            js = [x.decode() for x in JOINT.findall(d)]
            if js:
                joints.setdefault(w, set()).update(js)
        else:
            weapon_users[w].add(owner)
            tsf_weapons[owner].add(w)

print(f"scanned {pose_files} pose3d files\n")
print("=== weapon -> TSF that attach it in a pose")
for w in sorted(set(weapon_users) | set(weapon_poses)):
    us = sorted(weapon_users.get(w, ()))
    print(f"  {w:<18} users={us}")

print("\n=== TSF -> weapons")
for t in sorted(tsf_weapons):
    print(f"  {t:<18} {sorted(tsf_weapons[t])}")

print("\n=== weapon joint names (from its own pose3d file)")
for w in sorted(joints):
    print(f"  {w:<18} {sorted(joints[w])}")

out = {
    "weapon_users": {k: sorted(v) for k, v in weapon_users.items()},
    "weapon_poses": {k: sorted(v) for k, v in weapon_poses.items()},
    "tsf_weapons": {k: sorted(v) for k, v in tsf_weapons.items()},
    "weapon_joints": {k: sorted(v) for k, v in joints.items()},
}
os.makedirs(os.path.join(WS, "tools"), exist_ok=True)
with open(os.path.join(WS, "tools", "weapon_usage.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\nwrote tools/weapon_usage.json")
