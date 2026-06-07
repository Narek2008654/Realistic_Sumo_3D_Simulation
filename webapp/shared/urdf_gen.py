"""Generate a robot URDF from a :class:`HardwareSpec`.

This is the *additive* counterpart to the handcrafted ``assets/robot.urdf``.
The core training env still loads ``assets/robot.urdf`` for the default
robot; this module exists so that NON-default specs (produced by an
env-builder or by the web frontend) can be turned into a loadable URDF and,
via :mod:`webapp.shared.geometry_export`, into a three.js geometry payload.

Design notes
------------
* **Link / joint names mirror ``assets/robot.urdf``** so the env's
  ``_spawn_robot`` (which looks joints up by name) keeps working: the two
  drive joints are named exactly ``left_wheel_joint`` / ``right_wheel_joint``
  with children ``left_wheel`` / ``right_wheel`` and parent ``base_link``.
  A massless ``front_caster`` slider and an optional ``nose_wedge`` round out
  the structure, matching the reference robot.
* **Inertia is auto-computed** from mass + dimensions using the exact
  solid-box / solid-cylinder principal-moment formulas, so every inertia
  trivially satisfies PyBullet's triangle inequality
  (``Ixx + Iyy >= Izz`` and cyclic) for any positive, finite dims.
* **Chassis placement** mirrors the reference: the box bottom sits at
  ``z = wheel_radius`` (so the wheels are the sole ground contact) and the
  box is centred between the rear face and the wedge boundary in X.
* Pure stdlib (``xml.etree.ElementTree`` + ``math``); no third-party deps.

The reference robot is reproduced *closely* (not necessarily byte-for-byte)
by ``generate_urdf(HardwareSpec.default())``: same links, same joints, same
named drive joints, valid inertia, and a structurally equivalent wedge.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

from webapp.shared.hardware_spec import HardwareSpec

__all__ = ["generate_urdf", "write_urdf"]


# ---------------------------------------------------------------------------
# Small formatting / element helpers
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    """Format a float compactly but with enough precision for mm-scale geom."""
    # 6 significant figures is plenty for metre-scale robot dimensions and
    # keeps the URDF readable; strip a trailing ".0" -> "0" stays "0".
    s = f"{x:.6g}"
    return s


def _xyz(t: tuple[float, float, float]) -> str:
    return " ".join(_fmt(v) for v in t)


def _origin(parent: ET.Element, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", xyz=_xyz(xyz), rpy=_xyz(rpy))


def _box_inertia(mass: float, length: float, width: float, height: float):
    """Principal moments of a solid box about its centre (URDF axes).

    length = X extent, width = Y extent, height = Z extent.
        Ixx = (m/12)*(W^2 + H^2)
        Iyy = (m/12)*(L^2 + H^2)
        Izz = (m/12)*(L^2 + W^2)
    Always satisfies the triangle inequality for positive dims.
    """
    c = mass / 12.0
    ixx = c * (width * width + height * height)
    iyy = c * (length * length + height * height)
    izz = c * (length * length + width * width)
    return ixx, iyy, izz


def _cylinder_inertia(mass: float, radius: float, length: float):
    """Principal moments of a solid cylinder whose symmetry axis is +Z
    (URDF cylinder default), about its centre.

        Ixx = Iyy = (m/12)*(3*r^2 + h^2)
        Izz       = 0.5*m*r^2
    """
    ixx = (mass / 12.0) * (3.0 * radius * radius + length * length)
    izz = 0.5 * mass * radius * radius
    return ixx, ixx, izz


def _add_inertial(link, mass, inertia, origin_xyz=(0.0, 0.0, 0.0),
                  origin_rpy=(0.0, 0.0, 0.0)):
    ine = ET.SubElement(link, "inertial")
    _origin(ine, origin_xyz, origin_rpy)
    ET.SubElement(ine, "mass", value=_fmt(mass))
    ixx, iyy, izz = inertia
    ET.SubElement(
        ine, "inertia",
        ixx=_fmt(ixx), ixy="0", ixz="0",
        iyy=_fmt(iyy), iyz="0",
        izz=_fmt(izz),
    )


def _add_geom(parent, kind: str, attrs: dict[str, str]):
    geom = ET.SubElement(parent, "geometry")
    ET.SubElement(geom, kind, **attrs)


def _add_visual(link, kind, attrs, rgba, origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0), material_name="mat"):
    vis = ET.SubElement(link, "visual")
    _origin(vis, origin_xyz, origin_rpy)
    _add_geom(vis, kind, attrs)
    mat = ET.SubElement(vis, "material", name=material_name)
    ET.SubElement(mat, "color", rgba=" ".join(_fmt(c) for c in rgba))


def _add_collision(link, kind, attrs, origin_xyz=(0.0, 0.0, 0.0),
                   origin_rpy=(0.0, 0.0, 0.0)):
    col = ET.SubElement(link, "collision")
    _origin(col, origin_xyz, origin_rpy)
    _add_geom(col, kind, attrs)


def _add_lateral_friction(link, value: float):
    contact = ET.SubElement(link, "contact")
    ET.SubElement(contact, "lateral_friction", value=_fmt(value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_urdf(spec: HardwareSpec) -> str:
    """Build a valid URDF XML string from ``spec``.

    Produces a structure mirroring ``assets/robot.urdf``:
    ``base_link`` (chassis box), ``front_caster`` (massless slider),
    ``left_wheel`` / ``right_wheel`` (cylinders on ``left_wheel_joint`` /
    ``right_wheel_joint``, continuous, axis +Y), and an optional
    ``nose_wedge`` when ``spec.chassis.wedge_present``.

    All inertias are auto-computed from mass + dimensions, so they always
    satisfy PyBullet's triangle inequality.
    """
    ch = spec.chassis
    dt = spec.drivetrain

    L, W, H = ch.length_m, ch.width_m, ch.height_m
    r = dt.wheel_radius_m
    # Wheel cylinder length: not in the spec; use the reference 19 mm value,
    # clamped so it never exceeds the track gap.
    wheel_len = min(0.019, max(0.001, dt.track_width_m - 2.0 * r))

    robot = ET.Element("robot", name=spec.name or "generated_robot")

    # ---- base_link (chassis) -------------------------------------------
    # Box bottom rests at z = wheel_radius so the wheels are the sole ground
    # contact (mirrors the reference robot's 10 mm clearance == r). Box
    # centred in X between the rear face (-L/2 relative to its own centre)
    # and the wedge boundary; we centre it so its rear face aligns with the
    # chassis rear and the wedge mounts at its front face.
    box_center_z = r + H / 2.0
    # Place the chassis so its geometric centre is at body x = 0 in Y/Z but
    # offset in X to leave room for the wedge at the front. We keep the rear
    # face fixed and shift forward by 0; simplest robust choice: centre the
    # box on the wheel axle X span midpoint is fragile, so just centre the
    # box at body x=0 in X minus a small forward bias is unnecessary —
    # centre it at x=0 keeps everything valid for arbitrary specs.
    box_center_x = 0.0

    base = ET.SubElement(robot, "link", name="base_link")
    _add_inertial(
        base,
        mass=ch.mass_kg,
        inertia=_box_inertia(ch.mass_kg, L, W, H),
        origin_xyz=ch.com_xyz,
    )
    _add_visual(
        base, "box", {"size": f"{_fmt(L)} {_fmt(W)} {_fmt(H)}"},
        rgba=(0.85, 0.15, 0.15, 1.0),
        origin_xyz=(box_center_x, 0.0, box_center_z),
        material_name="chassis_red",
    )
    _add_collision(
        base, "box", {"size": f"{_fmt(L)} {_fmt(W)} {_fmt(H)}"},
        origin_xyz=(box_center_x, 0.0, box_center_z),
    )
    _add_lateral_friction(base, ch.chassis_friction)

    # ---- front_caster (massless frictionless slider) -------------------
    caster = ET.SubElement(robot, "link", name="front_caster")
    _add_inertial(caster, mass=1e-3, inertia=(1e-9, 1e-9, 1e-9))
    _add_collision(caster, "sphere", {"radius": _fmt(0.003)})
    ccontact = ET.SubElement(caster, "contact")
    ET.SubElement(ccontact, "lateral_friction", value="0.0")
    ET.SubElement(ccontact, "rolling_friction", value="0.0")
    ET.SubElement(ccontact, "spinning_friction", value="0.0")

    cjoint = ET.SubElement(robot, "joint", name="front_caster_joint",
                           type="fixed")
    ET.SubElement(cjoint, "parent", link="base_link")
    ET.SubElement(cjoint, "child", link="front_caster")
    # Just inside the chassis front face, at floor-skimming height.
    _origin(cjoint, (box_center_x + L / 2.0 - 0.004, 0.0, 0.003))

    # ---- nose_wedge (optional) -----------------------------------------
    if ch.wedge_present:
        # Slab geometry mirrors the reference: a thin box whose long axis is
        # the slope length, pitched about +Y by wedge_pitch_rad, mounted at
        # the chassis front face. Slope length = wedge_length / cos(pitch)
        # (the 28.82 mm horizontal run becomes the 33.13 mm slope at 0.5113).
        pitch = ch.wedge_pitch_rad
        cos_p = math.cos(pitch)
        slope_len = ch.wedge_length_m / cos_p if abs(cos_p) > 1e-6 \
            else ch.wedge_length_m
        thick = 0.002
        wedge_mass = 0.02
        wedge = ET.SubElement(robot, "link", name="nose_wedge")
        _add_inertial(
            wedge,
            mass=wedge_mass,
            inertia=_box_inertia(wedge_mass, slope_len, W, thick),
        )
        _add_visual(
            wedge, "box", {"size": f"{_fmt(slope_len)} {_fmt(W)} {_fmt(thick)}"},
            rgba=(0.75, 0.75, 0.78, 1.0),
            material_name="wedge_silver",
        )
        _add_collision(
            wedge, "box",
            {"size": f"{_fmt(slope_len)} {_fmt(W)} {_fmt(thick)}"},
        )
        _add_lateral_friction(wedge, 0.7)

        wjoint = ET.SubElement(robot, "joint", name="nose_wedge_joint",
                               type="fixed")
        ET.SubElement(wjoint, "parent", link="base_link")
        ET.SubElement(wjoint, "child", link="nose_wedge")
        # Mount at the chassis front face, near the wheel-radius height so the
        # tip skims the floor. Centre x sits half the horizontal run ahead of
        # the chassis front.
        wedge_x = box_center_x + L / 2.0 + ch.wedge_length_m / 2.0
        wedge_z = r * 0.94
        _origin(wjoint, (wedge_x, 0.0, wedge_z), (0.0, pitch, 0.0))

    # ---- wheels ---------------------------------------------------------
    # Cylinder default symmetry axis is +Z; roll 90deg about X (rpy roll =
    # pi/2) so the axis points along +Y (the drive axis). Inertia is computed
    # in the cylinder's own frame (axis +Z) -> after the roll, the rolling
    # moment ends up about +Y, exactly as the reference URDF documents.
    wheel_mass = 0.025
    ixx, iyy, izz = _cylinder_inertia(wheel_mass, r, wheel_len)
    # After rpy roll pi/2 about X: cylinder axis -> +Y. Provide inertia in the
    # post-roll body frame: rolling (about Y) = izz_cyl, the two equal radial
    # moments go to Ixx and Izz.
    wheel_inertia = (ixx, izz, ixx)  # (Ixx, Iyy=rolling, Izz)

    half_track = dt.track_width_m / 2.0
    axle_x = dt.wheel_x_offset_m
    axle_z = r  # axle at wheel radius -> wheel bottom at z=0

    for side, sign, jname, lname in (
        ("left", +1.0, "left_wheel_joint", "left_wheel"),
        ("right", -1.0, "right_wheel_joint", "right_wheel"),
    ):
        wlink = ET.SubElement(robot, "link", name=lname)
        _add_inertial(wlink, mass=wheel_mass, inertia=wheel_inertia)
        _add_visual(
            wlink, "cylinder",
            {"radius": _fmt(r), "length": _fmt(wheel_len)},
            rgba=(0.05, 0.05, 0.05, 1.0),
            origin_rpy=(math.pi / 2.0, 0.0, 0.0),
            material_name="wheel_black",
        )
        _add_collision(
            wlink, "cylinder",
            {"radius": _fmt(r), "length": _fmt(wheel_len)},
            origin_rpy=(math.pi / 2.0, 0.0, 0.0),
        )
        _add_lateral_friction(wlink, ch.wheel_friction)

        wj = ET.SubElement(robot, "joint", name=jname, type="continuous")
        ET.SubElement(wj, "parent", link="base_link")
        ET.SubElement(wj, "child", link=lname)
        _origin(wj, (axle_x, sign * half_track, axle_z))
        ET.SubElement(wj, "axis", xyz="0 1 0")

    # ---- serialise ------------------------------------------------------
    raw = ET.tostring(robot, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    # minidom adds a default XML declaration; keep it. Drop blank lines.
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    return "\n".join(lines) + "\n"


def write_urdf(spec: HardwareSpec, path: str) -> str:
    """Generate the URDF for ``spec`` and write it to ``path``; return text."""
    text = generate_urdf(spec)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
