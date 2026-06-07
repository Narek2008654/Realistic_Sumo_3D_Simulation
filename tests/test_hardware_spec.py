"""Unit tests for the HardwareSpec configuration contract.

Runs under pytest, but also under plain ``python tests/test_hardware_spec.py``
when pytest isn't installed (see the ``__main__`` runner at the bottom).

The "real" constants asserted here are read from sumo_env.py / obs_stack.py /
assets/robot.urdf; if HardwareSpec.default() drifts from them, these fail.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

# Make the repo root importable when run directly (tests/ -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from webapp.shared.hardware_spec import HardwareSpec  # noqa: E402


def test_obs_dim_is_21():
    assert HardwareSpec.default().obs_dim == 21


def test_action_dim_is_9():
    assert HardwareSpec.default().action_dim == 9


def test_base_obs_dim_is_12():
    # 3 raw distances + 9 engineered = obs_stack.BASE_OBS_DIM.
    assert HardwareSpec.default().base_obs_dim == 12


def test_n_distance_and_n_line():
    spec = HardwareSpec.default()
    assert spec.n_distance == 3
    assert spec.n_line == 2


def test_signature_round_trip_stable():
    """from_dict(to_dict()) must reproduce the identical signature."""
    spec = HardwareSpec.default()
    rebuilt = HardwareSpec.from_dict(spec.to_dict())
    assert rebuilt.obs_signature_hash == spec.obs_signature_hash


def test_dict_round_trip_equal():
    spec = HardwareSpec.default()
    assert HardwareSpec.from_dict(spec.to_dict()) == spec


def test_json_round_trip_equal():
    spec = HardwareSpec.default()
    assert HardwareSpec.from_json(spec.to_json()) == spec


def test_distance_sensor_angles_and_range_match_env():
    """Angles/range must match sumo_env.py _raw_distances + ENEMY_FAR_DIST."""
    spec = HardwareSpec.default()
    by_id = {s.id: s for s in spec.distance_sensors}
    assert set(by_id) == {"front", "left", "right"}

    # ENEMY_FAR_DIST = 0.80 (sumo_env.py:242).
    for s in spec.distance_sensors:
        assert s.range_m == 0.80
        # noise_sigma = DR_TOF_NOISE_SIGMA_PCT (0.02) * ENEMY_FAR_DIST.
        assert math.isclose(s.noise_sigma, 0.02 * 0.80)

    # front yaw+0, left yaw+30deg, right yaw-30deg (sumo_env.py:988-1005).
    assert by_id["front"].angle_rad == 0.0
    assert math.isclose(by_id["left"].angle_rad, math.radians(30.0))
    assert math.isclose(by_id["right"].angle_rad, math.radians(-30.0))


def test_discrete_grid_matches_env_action_map():
    spec = HardwareSpec.default()
    expected = (
        (-1.0, -1.0), (-1.0, 0.0), (-1.0, +1.0),
        (0.0, -1.0), (0.0, 0.0), (0.0, +1.0),
        (+1.0, -1.0), (+1.0, 0.0), (+1.0, +1.0),
    )
    assert spec.action_space.kind == "discrete"
    assert spec.action_space.grid == expected


def test_engineered_layout_matches_env():
    spec = HardwareSpec.default()
    assert spec.engineered == (
        "last_seen_dir", "line_l", "line_r", "prev_left", "prev_right",
        "engagement", "yaw_rate_proxy", "front_ir_delta", "lateral_ir_delta",
    )


def test_signature_deterministic_across_processes():
    """Spawn a fresh interpreter and confirm the hash is identical.

    Proves no hash randomization (PYTHONHASHSEED-independent). Uses a
    randomized seed in the child to make the point.
    """
    expected = HardwareSpec.default().obs_signature_hash
    code = (
        "from webapp.shared.hardware_spec import HardwareSpec;"
        "print(HardwareSpec.default().obs_signature_hash)"
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "12345"
    out = subprocess.check_output(
        [sys.executable, "-c", code], cwd=_REPO_ROOT, env=env, text=True,
    )
    assert out.strip() == expected


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
    spec = HardwareSpec.default()
    print(
        f"\nobs_signature_hash={spec.obs_signature_hash} "
        f"obs_dim={spec.obs_dim} action_dim={spec.action_dim}"
    )
    if failures:
        print(f"\n{failures} test(s) FAILED")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
