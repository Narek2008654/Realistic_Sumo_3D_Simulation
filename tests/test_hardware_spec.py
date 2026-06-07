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


# The default obs/action contract hash is FROZEN: every committed 21/9
# checkpoint is finetune-compatible only against this exact signature. Adding
# world geometry (dohyo size) or restructuring the wedge MUST NOT change it.
DEFAULT_OBS_SIGNATURE = "1878ebc3a009"


def test_obs_dim_is_21():
    assert HardwareSpec.default().obs_dim == 21


def test_default_obs_signature_unchanged():
    """The default signature is pinned (dohyo + wedge changes excluded)."""
    assert HardwareSpec.default().obs_signature_hash == DEFAULT_OBS_SIGNATURE


def test_dohyo_defaults_present():
    """Default ring radius/border match sumo_env DOHYO_RADIUS / BORDER_WIDTH."""
    dohyo = HardwareSpec.default().dohyo
    assert math.isclose(dohyo.radius_m, 0.35)        # DOHYO_RADIUS = 0.70/2
    assert math.isclose(dohyo.border_width_m, 0.025)  # BORDER_WIDTH
    # Inner (black-disc) radius is radius - border = 0.325 (INNER_RADIUS).
    assert math.isclose(dohyo.radius_m - dohyo.border_width_m, 0.325)


def test_dohyo_excluded_from_signature():
    """Changing the dohyo radius must NOT change the obs signature."""
    import dataclasses

    base = HardwareSpec.default()
    big_ring = dataclasses.replace(base, dohyo=dataclasses.replace(
        base.dohyo, radius_m=0.50))
    assert big_ring.obs_signature_hash == base.obs_signature_hash
    assert big_ring.obs_signature_hash == DEFAULT_OBS_SIGNATURE


def test_wedge_pitch_derived_matches_today():
    """Derived pitch reproduces the historical 0.5113 rad wedge."""
    ch = HardwareSpec.default().chassis
    assert ch.wedge_low_height_m == 0.0
    # high = run * tan(0.5113) so the derived pitch is exactly 0.5113.
    assert math.isclose(ch.wedge_pitch_rad, 0.5113, abs_tol=1e-12)
    # body_length_m is an alias of length_m.
    assert ch.body_length_m == ch.length_m


def test_wedge_low_high_round_trip():
    """Low/high edge heights survive dict + JSON round-trips."""
    spec = HardwareSpec.default()
    for rebuilt in (HardwareSpec.from_dict(spec.to_dict()),
                    HardwareSpec.from_json(spec.to_json())):
        assert rebuilt.chassis.wedge_low_height_m == spec.chassis.wedge_low_height_m
        assert rebuilt.chassis.wedge_high_height_m == spec.chassis.wedge_high_height_m
        assert math.isclose(rebuilt.chassis.wedge_pitch_rad,
                            spec.chassis.wedge_pitch_rad)
        assert rebuilt.dohyo == spec.dohyo


def test_wedge_pitch_override():
    """An explicit override wins over the derived pitch."""
    import dataclasses

    base = HardwareSpec.default()
    forced = dataclasses.replace(
        base.chassis, wedge_pitch_override_rad=1.0)
    assert forced.wedge_pitch_rad == 1.0


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
