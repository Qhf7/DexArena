"""Derive the single-arm gripper URDFs from the dual-arm one.

    panda_gripper_left_mobile_base.urdf    left arm at y = +0.22
    panda_gripper_right_mobile_base.urdf   right arm at y = -0.22

Both keep the full-size trolley of ``panda_dual_gripper_mobile_base.urdf`` and leave the
arm at its dual mount, so each single arm is the same arm, joint for joint, as the
matching arm of the dual robot (the rule the Sharpa, Wuji and XHand singles follow).
The other arm is pruned and the kept chain loses its side prefix
(``right_panda_link3`` -> ``panda_link3``), so the stock Panda agent's joint, link and
TCP names apply unchanged.

Usage:
    python scripts/build_gripper_mobile_base_urdfs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_DIR = Path(__file__).resolve().parent.parent / "tasks" / "assets" / "urdf"
SRC = URDF_DIR / "panda_dual_gripper_mobile_base.urdf"
OUT = {
    "left": URDF_DIR / "panda_gripper_left_mobile_base.urdf",
    "right": URDF_DIR / "panda_gripper_right_mobile_base.urdf",
}


def _strip(name: str, side: str) -> str:
    prefix = f"{side}_"
    if name == f"mobile_base_to_{side}_panda":
        return "mobile_base_to_panda"
    return name[len(prefix):] if name.startswith(prefix) else name


def build(side: str) -> ET.ElementTree:
    other = "right" if side == "left" else "left"
    root = copy.deepcopy(ET.parse(SRC).getroot())
    root.set("name", f"panda_gripper_{side}_mobile_base")
    for element in list(root):
        name = element.get("name", "")
        drop = name.startswith(f"{other}_") or name == f"mobile_base_to_{other}_panda"
        if drop:
            root.remove(element)
            continue
        element.set("name", _strip(name, side))
        for ref in element.iter():
            if ref.tag in ("parent", "child", "mimic") and ref.get("link") is not None:
                ref.set("link", _strip(ref.get("link"), side))
            if ref.tag == "mimic" and ref.get("joint") is not None:
                ref.set("joint", _strip(ref.get("joint"), side))
    return ET.ElementTree(root)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the link/joint counts and write nothing")
    args = ap.parse_args()
    for side, out in OUT.items():
        tree = build(side)
        root = tree.getroot()
        links = [e.get("name") for e in root.findall("link")]
        joints = [e.get("name") for e in root.findall("joint")]
        mount = next(j for j in root.findall("joint") if j.get("name") == "mobile_base_to_panda")
        print(f"{out.name}: {len(links)} links, {len(joints)} joints, mount origin {mount.find('origin').get('xyz')}")
        if not args.dry_run:
            ET.indent(tree, space="  ")
            tree.write(out, xml_declaration=True, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
