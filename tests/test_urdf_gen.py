"""Tests for the spec -> URDF -> geometry generators (feature E1e).

Runs under pytest, and under plain ``python tests/test_urdf_gen.py`` when
pytest isn't installed (``__main__`` runner at the bottom).

The load tests spin up a headless PyBullet DIRECT client, write the
generated URDF to a temp file, and assert it loads with no error, exposes
the named drive joints, and reports the chassis base mass from the spec.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

# Make the repo root importable when run directly (tests/ -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pybullet as p  # noqa: E402

from webapp.shared.geometry_export import (  # noqa: E402
    spec_to_geometry,
    urdf_to_geometry,
)
from webapp.shared.hardware_spec import HardwareSpec  # noqa: E402
from webapp.shared.urdf_gen import generate_urdf  # noqa: E402


# ---------------------------------------------------------------------------
# Spec variants
# ---------------------------------------------------------------------------
def _heavy_spec() -> HardwareSpec:
    """A heavier 0.9 kg chassis with bigger wheels/track."""
    base = HardwareSpec.default()
    chassis = dataclasses.replace(
        base.chassis, mass_kg=0.9, length_m=0.090, width_m=0.110,
        height_m=0.060,
    )
    drivetrain = dataclasses.replace(
        base.drivetrain, wheel_radius_m=0.020, track_width_m=0.120,
        wheel_x_offset_m=-0.040,
    )
    return dataclasses.replace(
        base, name="heavy", chassis=chassis, drivetrain=drivetrain,
    )


def _no_wedge_spec() -> HardwareSpec:
    base = HardwareSpec.default()
    chassis = dataclasses.replace(base.chassis, wedge_present=False)
    return dataclasses.replace(base, name="no_wedge", chassis=chassis)


# ---------------------------------------------------------------------------
# PyBullet load helper
# ---------------------------------------------------------------------------
def _load_in_pybullet(spec: HardwareSpec):
    """Load ``generate_urdf(spec)`` in a DIRECT client; return (cid, body)."""
    urdf = generate_urdf(spec)
    cid = p.connect(p.DIRECT)
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".urdf", delete=False, encoding="utf-8",
        ) as fh:
            fh.write(urdf)
            path = fh.name
        try:
            body = p.loadURDF(path, physicsClientId=cid)
        finally:
            os.unlink(path)
    except Exception:
        p.disconnect(cid)
        raise
    return cid, body


def _joint_names(cid: int, body: int) -> set[str]:
    n = p.getNumJoints(body, physicsClientId=cid)
    names = set()
    for j in range(n):
        info = p.getJointInfo(body, j, physicsClientId=cid)
        names.add(info[1].decode("utf-8"))
    return names


# ---------------------------------------------------------------------------
# Tests: loading
# ---------------------------------------------------------------------------
def test_default_spec_loads_with_named_joints():
    spec = HardwareSpec.default()
    cid, body = _load_in_pybullet(spec)
    try:
        names = _joint_names(cid, body)
        assert "left_wheel_joint" in names
        assert "right_wheel_joint" in names
        # Base link mass should match the chassis spec mass (within rounding).
        base_mass = p.getDynamicsInfo(body, -1, physicsClientId=cid)[0]
        assert abs(base_mass - spec.chassis.mass_kg) < 1e-3
    finally:
        p.disconnect(cid)


def test_heavy_spec_loads():
    spec = _heavy_spec()
    cid, body = _load_in_pybullet(spec)
    try:
        names = _joint_names(cid, body)
        assert "left_wheel_joint" in names
        assert "right_wheel_joint" in names
        base_mass = p.getDynamicsInfo(body, -1, physicsClientId=cid)[0]
        assert abs(base_mass - spec.chassis.mass_kg) < 1e-3
    finally:
        p.disconnect(cid)


def test_no_wedge_spec_loads():
    spec = _no_wedge_spec()
    cid, body = _load_in_pybullet(spec)
    try:
        names = _joint_names(cid, body)
        assert "left_wheel_joint" in names
        assert "right_wheel_joint" in names
        # No nose_wedge link/joint expected.
        assert "nose_wedge_joint" not in names
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------------------
# Tests: geometry export
# ---------------------------------------------------------------------------
def test_geometry_export_default():
    geo = spec_to_geometry(HardwareSpec.default())
    assert "links" in geo and "joints" in geo
    # chassis + 2 wheels + wedge = 4 visual links (caster has no visual).
    assert len(geo["links"]) >= 3

    valid_shapes = {"box", "cylinder", "sphere", "mesh"}
    for link in geo["links"]:
        assert link["shape"] in valid_shapes
        assert isinstance(link["origin_xyz"], list) and len(link["origin_xyz"]) == 3
        assert isinstance(link["origin_rpy"], list) and len(link["origin_rpy"]) == 3
        if link["shape"] == "box":
            assert len(link["size"]) == 3
        elif link["shape"] == "cylinder":
            assert link["radius"] > 0 and link["length"] > 0
        assert len(link["rgba"]) == 4

    jnames = {j["name"] for j in geo["joints"]}
    assert "left_wheel_joint" in jnames
    assert "right_wheel_joint" in jnames
    # Wheel joints carry an axis and the right parent/child.
    by_name = {j["name"]: j for j in geo["joints"]}
    for jn in ("left_wheel_joint", "right_wheel_joint"):
        assert by_name[jn]["parent"] == "base_link"
        assert by_name[jn]["axis"] == [0.0, 1.0, 0.0]


def test_geometry_export_mesh_passthrough():
    urdf = (
        '<robot name="m">'
        '<link name="base_link"><visual><geometry>'
        '<mesh filename="foo.stl" scale="1 1 1"/>'
        '</geometry></visual></link>'
        '</robot>'
    )
    geo = urdf_to_geometry(urdf)
    assert geo["links"][0]["shape"] == "mesh"
    assert geo["links"][0]["filename"] == "foo.stl"


# ---------------------------------------------------------------------------
# Plain-python runner (no pytest)
# ---------------------------------------------------------------------------
def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    if failures:
        print(f"\n{failures} test(s) FAILED")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
