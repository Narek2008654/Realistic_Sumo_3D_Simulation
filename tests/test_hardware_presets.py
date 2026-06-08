"""Tests for the hardware-preset library.

Runnable two ways (mirrors the other backend suites)::

    python tests/test_hardware_presets.py   # plain-python harness (PASS/FAIL)
    pytest tests/test_hardware_presets.py     # standard pytest

Asserts every preset round-trips through ``HardwareSpec.from_dict`` and that the
faithful NovaMax preset matches the URDF + ``NOVAMAX_*`` constants. Pure: no
PyBullet, no network — just spec construction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.shared.hardware_presets import (  # noqa: E402
    PRESET_IDS,
    get_preset,
    list_presets,
)
from webapp.shared.hardware_spec import (  # noqa: E402
    HardwareSpec,
    mini_sumo_violations,
)

# Faithful-NovaMax reference numbers (assets/novamax.urdf + sumo_env NOVAMAX_*).
_NOVAMAX_MAX_RAD = (800.0 / 60.0) * 2.0 * math.pi   # 83.776 rad/s, NOVAMAX_MAX_RAD
_NOVAMAX_MAX_FORCE = 0.50                            # NOVAMAX_MAX_FORCE
_NOVAMAX_TOTAL_MASS = 0.45                           # 0.38 + 0.02 + 2*0.025
_NOVAMAX_PITCH = 0.764                               # plow_joint rpy


def test_all_presets_present() -> None:
    assert set(PRESET_IDS) == {
        "novamax", "wedge_pusher", "speedster", "heavy_rammer", "spinner_disc",
    }, PRESET_IDS


def test_every_preset_round_trips() -> None:
    presets = list_presets()
    assert len(presets) == len(PRESET_IDS)
    for p in presets:
        assert set(p) >= {"id", "name", "description", "hardware_spec"}, p
        # Round-trip: from_dict must build, and re-serialise identically.
        spec = HardwareSpec.from_dict(p["hardware_spec"])
        assert isinstance(spec, HardwareSpec)
        assert spec.to_dict() == p["hardware_spec"], p["id"]
        # Contract preserved (default obs/action signature) so a preset stays
        # finetune-compatible with existing checkpoints.
        assert spec.obs_signature_hash == HardwareSpec.default().obs_signature_hash
        # Physical sanity.
        assert spec.chassis.mass_kg > 0
        assert spec.drivetrain.max_omega_rad_s > 0
        assert spec.drivetrain.max_torque_nm > 0


def test_all_presets_mini_sumo_legal() -> None:
    """Every shipped preset must obey the mini-sumo class (<=500 g, <=10x10 cm),
    so it can be saved as a robot/opponent without tripping the gate."""
    for p in list_presets():
        spec = HardwareSpec.from_dict(p["hardware_spec"])
        assert mini_sumo_violations(spec) == [], (p["id"], mini_sumo_violations(spec))


def test_novamax_is_faithful() -> None:
    spec = HardwareSpec.from_dict(get_preset("novamax")["hardware_spec"])
    c, d = spec.chassis, spec.drivetrain
    # Chassis box + mass + CoM from the URDF.
    assert (c.length_m, c.width_m, c.height_m) == (0.099, 0.090, 0.035)
    assert c.mass_kg == _NOVAMAX_TOTAL_MASS
    assert c.com_xyz == (-0.015, 0.0, 0.010)
    # Drivetrain from URDF geometry + NOVAMAX_* motor caps.
    assert d.wheel_radius_m == 0.0165
    assert d.track_width_m == 0.068
    assert d.wheel_x_offset_m == -0.030
    assert abs(d.max_omega_rad_s - _NOVAMAX_MAX_RAD) < 1e-2
    assert d.max_torque_nm == _NOVAMAX_MAX_FORCE
    # Wedge pitch reproduces the plow joint's 0.764 rad.
    assert abs(c.wedge_pitch_rad - _NOVAMAX_PITCH) < 1e-3


def _main() -> int:
    tests = [
        ("all_presets_present", test_all_presets_present),
        ("every_preset_round_trips", test_every_preset_round_trips),
        ("all_presets_mini_sumo_legal", test_all_presets_mini_sumo_legal),
        ("novamax_is_faithful", test_novamax_is_faithful),
    ]
    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
            print(f"  FAIL  {name}: {exc!r}")
    print(f"\nPASS: {len(passed)} / FAIL: {len(failed)}")
    for name, err in failed:
        print(f"    - {name}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_main())
