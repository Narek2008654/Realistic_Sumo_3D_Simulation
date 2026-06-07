"""Tests for the spec-driven dohyo (ring) geometry in MiniSumoEnv.

The dohyo radius / border are now read from ``HardwareSpec.dohyo``. With the
default spec the env must be byte-identical to today (the heavy lifting is done
by ``test_dynamics_faithfulness_spec`` / ``test_obs_faithfulness_spec``); here
we just assert:

  * the default spec maps onto the historical module constants exactly, and
  * a non-default ``dohyo.radius_m`` actually changes the ring the env builds
    (radius / inner / spawn attrs scale, and an episode can end differently).

Runs under pytest and under plain ``python tests/test_dohyo_spec.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import sys

# Import torch FIRST on Windows (DLL-search-order race vs numpy); pulling in
# sumo_env transitively drags in the torch-backed stack.
import torch  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dataclasses  # noqa: E402

import sumo_env  # noqa: E402
from sumo_env import (  # noqa: E402
    BORDER_WIDTH,
    DOHYO_RADIUS,
    INNER_RADIUS,
    SPAWN_RADIUS,
    MiniSumoEnv,
)
from webapp.shared.hardware_spec import HardwareSpec  # noqa: E402


def test_default_spec_ring_matches_module_constants():
    """Default spec => instance ring attrs equal the historical constants."""
    env = MiniSumoEnv(hardware_spec=HardwareSpec.default())
    try:
        assert env._dohyo_radius == DOHYO_RADIUS
        assert env._border_width == BORDER_WIDTH
        assert env._inner_radius == INNER_RADIUS
        assert env._spawn_radius == SPAWN_RADIUS
    finally:
        env.close()


def test_default_spec_ring_matches_when_spec_is_none():
    """None spec defaults to HardwareSpec.default() => same ring attrs."""
    env = MiniSumoEnv()
    try:
        assert env._dohyo_radius == DOHYO_RADIUS
        assert env._inner_radius == INNER_RADIUS
        assert env._spawn_radius == SPAWN_RADIUS
    finally:
        env.close()


def test_nondefault_radius_scales_the_ring():
    """A bigger dohyo.radius_m scales the platform, inner ring and spawn."""
    base = HardwareSpec.default()
    big = dataclasses.replace(
        base, dohyo=dataclasses.replace(base.dohyo, radius_m=0.50))
    env = MiniSumoEnv(hardware_spec=big)
    try:
        assert env._dohyo_radius == 0.50
        # inner = radius - border; border unchanged.
        assert math.isclose(env._inner_radius, 0.50 - BORDER_WIDTH)
        # spawn scales with the ring fraction (0.50/0.35).
        assert math.isclose(env._spawn_radius,
                            SPAWN_RADIUS * (0.50 / DOHYO_RADIUS))
        assert env._spawn_radius > SPAWN_RADIUS
        # The spec is faithfully stored on the env.
        assert env.hw_spec.dohyo.radius_m == 0.50
    finally:
        env.close()


def test_nondefault_radius_builds_and_resets():
    """The env with a different ring builds + resets without error."""
    base = HardwareSpec.default()
    small = dataclasses.replace(
        base, dohyo=dataclasses.replace(base.dohyo, radius_m=0.25))
    env = MiniSumoEnv(hardware_spec=small)
    try:
        obs, info = env.reset(seed=0)
        # reset returns the single-frame (base) observation; the frame stack
        # is assembled by the training wrapper.
        assert obs.shape[0] == env.hw_spec.base_obs_dim
        assert env._dohyo_radius == 0.25
    finally:
        env.close()


def test_module_constants_still_defined():
    """Back-compat: the module constants the audit imports still exist."""
    assert sumo_env.DOHYO_RADIUS == 0.35
    assert sumo_env.INNER_RADIUS == 0.35 - 0.025
    assert sumo_env.SPAWN_RADIUS == 0.25


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
