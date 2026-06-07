"""Convert a URDF into a flat, three.js-friendly geometry description.

The frontend renders the robot from primitives (box / cylinder / sphere),
so we parse a URDF (the handcrafted one or one from :mod:`urdf_gen`) into a
JSON-able dict that maps cleanly onto three.js mesh construction:

    {
      "links": [
        {"name", "shape", "size"|("radius","length")|"radius",
         "origin_xyz", "origin_rpy", "rgba"},
        ...
      ],
      "joints": [
        {"name", "parent", "child", "origin_xyz", "origin_rpy", "axis"},
        ...
      ],
    }

Only the *visual* geometry of each link is emitted (one entry per link that
has a ``<visual>``; links without a visual, e.g. the front caster, are
skipped). ``origin_xyz`` / ``origin_rpy`` are the visual element's local
origin; the frontend composes them with the joint transforms to place each
mesh. Box -> ``size`` (3-list). Cylinder -> ``radius`` + ``length``.
Sphere -> ``radius``. A ``<mesh>`` is passed through as
``{"shape": "mesh", "filename": ...}`` (no file loading here).

Pure stdlib (``xml.etree.ElementTree``); no third-party deps.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from webapp.shared.hardware_spec import HardwareSpec
from webapp.shared.urdf_gen import generate_urdf

__all__ = ["urdf_to_geometry", "spec_to_geometry"]

_ZERO3 = [0.0, 0.0, 0.0]


def _parse_vec(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    return [float(p) for p in text.split()]


def _origin_of(elem: ET.Element | None) -> tuple[list[float], list[float]]:
    """Return (xyz, rpy) of a child ``<origin>`` (zeros if absent)."""
    if elem is None:
        return list(_ZERO3), list(_ZERO3)
    origin = elem.find("origin")
    if origin is None:
        return list(_ZERO3), list(_ZERO3)
    return (
        _parse_vec(origin.get("xyz"), _ZERO3),
        _parse_vec(origin.get("rpy"), _ZERO3),
    )


def _rgba_of(visual: ET.Element) -> list[float]:
    """Extract the visual material's rgba; default opaque grey."""
    mat = visual.find("material")
    if mat is not None:
        color = mat.find("color")
        if color is not None and color.get("rgba"):
            return _parse_vec(color.get("rgba"), [0.6, 0.6, 0.6, 1.0])
    return [0.6, 0.6, 0.6, 1.0]


def _shape_of(geometry: ET.Element) -> dict[str, Any]:
    """Map a URDF ``<geometry>`` child to a three.js-friendly shape dict."""
    box = geometry.find("box")
    if box is not None:
        return {"shape": "box", "size": _parse_vec(box.get("size"), _ZERO3)}

    cyl = geometry.find("cylinder")
    if cyl is not None:
        return {
            "shape": "cylinder",
            "radius": float(cyl.get("radius", "0")),
            "length": float(cyl.get("length", "0")),
        }

    sph = geometry.find("sphere")
    if sph is not None:
        return {"shape": "sphere", "radius": float(sph.get("radius", "0"))}

    mesh = geometry.find("mesh")
    if mesh is not None:
        out: dict[str, Any] = {"shape": "mesh", "filename": mesh.get("filename")}
        if mesh.get("scale"):
            out["scale"] = _parse_vec(mesh.get("scale"), [1.0, 1.0, 1.0])
        return out

    raise ValueError(
        "unsupported <geometry>: expected box/cylinder/sphere/mesh, got "
        f"children {[c.tag for c in geometry]!r}"
    )


def urdf_to_geometry(urdf_str: str) -> dict[str, Any]:
    """Parse a URDF XML string into a flat three.js geometry dict.

    See the module docstring for the output schema. Raises ``ValueError`` on
    malformed XML or an unsupported visual geometry.
    """
    try:
        root = ET.fromstring(urdf_str)
    except ET.ParseError as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"invalid URDF XML: {exc}") from exc

    links: list[dict[str, Any]] = []
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue  # collision-only links (e.g. front caster) aren't drawn
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        xyz, rpy = _origin_of(visual)
        entry: dict[str, Any] = {"name": link.get("name", "")}
        entry.update(_shape_of(geometry))
        entry["origin_xyz"] = xyz
        entry["origin_rpy"] = rpy
        entry["rgba"] = _rgba_of(visual)
        links.append(entry)

    joints: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        xyz, rpy = _origin_of(joint)
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        joints.append({
            "name": joint.get("name", ""),
            "type": joint.get("type", ""),
            "parent": parent.get("link") if parent is not None else None,
            "child": child.get("link") if child is not None else None,
            "origin_xyz": xyz,
            "origin_rpy": rpy,
            "axis": (
                _parse_vec(axis.get("xyz"), _ZERO3)
                if axis is not None else None
            ),
        })

    return {"links": links, "joints": joints}


def spec_to_geometry(spec: HardwareSpec) -> dict[str, Any]:
    """Convenience: ``urdf_to_geometry(generate_urdf(spec))``."""
    return urdf_to_geometry(generate_urdf(spec))
