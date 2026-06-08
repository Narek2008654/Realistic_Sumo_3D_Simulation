"""Hardware-preset library for custom opponents (and robots).

A *hardware preset* is a named, ready-to-use :class:`HardwareSpec` chassis the
UI can drop into the opponent/robot builder. Each preset crosses cleanly with
any behavior (zoo or DSL), so a user can save e.g. "Heavy Dodger" (the built-in
dodger behavior on the ``heavy_rammer`` chassis) without authoring geometry.

Every preset is derived from :meth:`HardwareSpec.default` so its
observation/action CONTRACT (sensor count, stack_k, action grid, engineered
features, reward schedule) is byte-compatible with today's robot — only the
PHYSICAL parameters (chassis box, mass, CoM, friction, wedge, drivetrain caps,
ring) differ. That keeps a preset spec loadable against existing checkpoints
and means every preset round-trips through :meth:`HardwareSpec.from_dict`.

Presets:

  * **novamax** — FAITHFUL to our reference enemy. Chassis dims/mass/CoM are
    read from ``assets/novamax.urdf``; the wedge from the plow joint; the
    drivetrain caps from the ``NOVAMAX_*`` constants in ``sumo_env.py``. See the
    mapping comments on :func:`_novamax` for the exact source of each number.
  * **wedge_pusher** / **speedster** / **heavy_rammer** / **spinner_disc** —
    clearly-original archetypes (hardware only; pair with any behavior).

Pure Python-3.12 stdlib + :class:`HardwareSpec` (itself stdlib-only) — no
third-party deps — so the backend router and tests can import it freely.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .hardware_spec import Chassis, Dohyo, Drivetrain, HardwareSpec

__all__ = ["list_presets", "get_preset", "PRESET_IDS"]


def _base() -> HardwareSpec:
    """Today's robot — the contract donor (sensors/action/reward/stack_k)."""
    return HardwareSpec.default()


def _spec(
    name: str,
    *,
    chassis: Chassis,
    drivetrain: Drivetrain,
    dohyo: Dohyo | None = None,
) -> HardwareSpec:
    """Clone the default spec, swapping only physical components.

    Keeps the default sensors, action grid, reward, engineered tuple and
    ``stack_k`` so the obs/action signature is unchanged (the preset stays
    finetune-compatible with existing checkpoints).
    """
    base = _base()
    return replace(
        base,
        name=name,
        chassis=chassis,
        drivetrain=drivetrain,
        dohyo=dohyo if dohyo is not None else base.dohyo,
    )


# ---------------------------------------------------------------------------
# novamax — FAITHFUL to assets/novamax.urdf + sumo_env.py NOVAMAX_* constants
# ---------------------------------------------------------------------------
def _novamax() -> HardwareSpec:
    """The reference NovaMax kit-bot as a HardwareSpec.

    Number-by-number mapping (source -> field):

      assets/novamax.urdf
        :40  chassis box "0.099 0.090 0.035"  -> length/width/height_m
        :27  mass value 0.38 (chassis link)   -+ total robot mass 0.45 kg
              + plow 0.02 (:92) + 2x wheel 0.025 (:143,:186) = 0.45  -> mass_kg
        :26  inertial origin "-0.015 0.0 0.010" -> com_xyz
        :136 plow_joint rpy "0 0.764 0"        -> wedge pitch 0.764 rad
        :104 plow box X 0.01386                -> wedge_length_m (run)
        :122 plow lateral_friction 0.7         (kept on chassis_friction-ish;
              chassis box itself has no <contact>, so we keep the default body
              friction and model the wedge separately via wedge_* below)
        :178 left_wheel_joint origin "-0.030 0.034 0.0165"
              -> wheel_x_offset_m -0.030 ; track_width_m 2*0.034 = 0.068
        :155 wheel cylinder radius 0.0165      -> wheel_radius_m

      sumo_env.py
        :155 NOVAMAX_MAX_RAD  = 83.776 rad/s (800 RPM)  -> max_omega_rad_s
        :156 NOVAMAX_MAX_FORCE = 0.50 N*m                -> max_torque_nm

    The wedge low/high edge heights are chosen so the DERIVED pitch reproduces
    the URDF's 0.764 rad over the 0.01386 m run: with the front lip on the floor
    (low = 0), high = run * tan(0.764).
    """
    base = _base()
    wedge_run = 0.01386                         # novamax.urdf:104 plow box X
    wedge_low = 0.0
    wedge_high = wedge_run * math.tan(0.764)    # -> derived pitch 0.764 rad

    chassis = replace(
        base.chassis,
        length_m=0.099,                         # novamax.urdf:40 box X
        width_m=0.090,                          # novamax.urdf:40 box Y
        height_m=0.035,                         # novamax.urdf:40 box Z
        mass_kg=0.45,                           # total: 0.38 + 0.02 + 2*0.025
        com_xyz=(-0.015, 0.0, 0.010),           # novamax.urdf:26 inertial origin
        wedge_present=True,
        wedge_length_m=wedge_run,
        wedge_low_height_m=wedge_low,
        wedge_high_height_m=wedge_high,
        wedge_pitch_override_rad=None,          # derived 0.764 from low/high/run
    )
    drivetrain = replace(
        base.drivetrain,
        wheel_radius_m=0.0165,                  # novamax.urdf:155 cylinder radius
        track_width_m=0.068,                    # novamax.urdf:178 axle y +/-0.034
        wheel_x_offset_m=-0.030,                # novamax.urdf:178 axle x
        max_torque_nm=0.50,                     # sumo_env.py NOVAMAX_MAX_FORCE
        max_omega_rad_s=83.776,                 # sumo_env.py NOVAMAX_MAX_RAD
    )
    return _spec("novamax", chassis=chassis, drivetrain=drivetrain)


# ---------------------------------------------------------------------------
# Original archetypes (hardware only — pair with any behavior)
# ---------------------------------------------------------------------------
def _wedge_pusher() -> HardwareSpec:
    """Low, wide wedge with a long sharp plow and high push torque."""
    base = _base()
    wedge_run = 0.040
    chassis = replace(
        base.chassis,
        length_m=0.080,
        width_m=0.100,                          # wide (at the 10 cm class limit)
        height_m=0.030,                         # low
        mass_kg=0.50,                           # at the 500 g class limit
        com_xyz=(-0.010, 0.0, 0.006),           # low CoM, slightly rear
        wedge_present=True,
        wedge_length_m=wedge_run,               # long plow
        wedge_low_height_m=0.0,                  # tip skims the floor
        wedge_high_height_m=0.012,              # shallow ramp
        wedge_pitch_override_rad=None,
    )
    drivetrain = replace(
        base.drivetrain,
        wheel_radius_m=0.012,
        track_width_m=0.090,
        max_torque_nm=0.60,                     # strong pusher
        max_omega_rad_s=45.0,
    )
    return _spec("wedge_pusher", chassis=chassis, drivetrain=drivetrain)


def _speedster() -> HardwareSpec:
    """Light, compact, very fast — wins by maneuver, not by push."""
    base = _base()
    chassis = replace(
        base.chassis,
        length_m=0.070,
        width_m=0.075,
        height_m=0.040,
        mass_kg=0.28,                           # light
        com_xyz=(0.0, 0.0, 0.009),
        wedge_present=True,
        wedge_length_m=0.012,                   # token plow
        wedge_low_height_m=0.0,
        wedge_high_height_m=0.006,
        wedge_pitch_override_rad=None,
    )
    drivetrain = replace(
        base.drivetrain,
        wheel_radius_m=0.018,                   # big wheels -> high top speed
        track_width_m=0.062,
        max_torque_nm=0.18,
        max_omega_rad_s=95.0,                   # high omega
    )
    return _spec("speedster", chassis=chassis, drivetrain=drivetrain)


def _heavy_rammer() -> HardwareSpec:
    """Heavy, strong, no plow — wins head-on pushing contests by momentum."""
    base = _base()
    chassis = replace(
        base.chassis,
        length_m=0.100,
        width_m=0.100,
        height_m=0.055,
        mass_kg=0.50,                           # heavy — at the 500 g class limit
        com_xyz=(0.0, 0.0, 0.012),
        wedge_present=False,                    # no wedge — pure ram
        wedge_length_m=0.0,
        wedge_low_height_m=0.0,
        wedge_high_height_m=0.0,
        wedge_pitch_override_rad=None,
    )
    drivetrain = replace(
        base.drivetrain,
        wheel_radius_m=0.016,
        track_width_m=0.085,
        max_torque_nm=0.70,                     # strong
        max_omega_rad_s=55.0,
    )
    return _spec("heavy_rammer", chassis=chassis, drivetrain=drivetrain)


def _spinner_disc() -> HardwareSpec:
    """Compact, balanced, square footprint — agile mid-weight all-rounder."""
    base = _base()
    chassis = replace(
        base.chassis,
        length_m=0.085,
        width_m=0.085,                          # square
        height_m=0.045,
        mass_kg=0.45,                           # balanced mid-weight
        com_xyz=(0.0, 0.0, 0.010),              # centred CoM
        wedge_present=True,
        wedge_length_m=0.018,
        wedge_low_height_m=0.0,
        wedge_high_height_m=0.009,
        wedge_pitch_override_rad=None,
    )
    drivetrain = replace(
        base.drivetrain,
        wheel_radius_m=0.014,
        track_width_m=0.075,
        max_torque_nm=0.35,
        max_omega_rad_s=65.0,
    )
    return _spec("spinner_disc", chassis=chassis, drivetrain=drivetrain)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_PRESETS: dict[str, dict[str, Any]] = {
    "novamax": {
        "name": "NovaMax (faithful)",
        "description": (
            "The reference NovaMax kit-bot: steel chassis (99x90x35 mm, 0.45 kg "
            "total, low rear CoM), sharp ~44deg steel plow, and fast 800 RPM "
            "gearmotors (83.8 rad/s, 0.50 N*m). Derived from assets/novamax.urdf "
            "and the NOVAMAX_* constants."
        ),
        "factory": _novamax,
    },
    "wedge_pusher": {
        "name": "Wedge Pusher",
        "description": (
            "Low, wide stance with a long sharp plow and high push torque. "
            "Built to get under an opponent and shovel it out."
        ),
        "factory": _wedge_pusher,
    },
    "speedster": {
        "name": "Speedster",
        "description": (
            "Light and compact with big wheels and very high wheel speed. Wins "
            "by maneuvering and flanking rather than out-pushing."
        ),
        "factory": _speedster,
    },
    "heavy_rammer": {
        "name": "Heavy Rammer",
        "description": (
            "Heavy and strong with no wedge — a pure momentum brawler that wins "
            "head-on pushing contests."
        ),
        "factory": _heavy_rammer,
    },
    "spinner_disc": {
        "name": "Spinner Disc",
        "description": (
            "Compact square footprint, balanced mid-weight with a centred CoM — "
            "an agile all-rounder."
        ),
        "factory": _spinner_disc,
    },
}

PRESET_IDS: tuple[str, ...] = tuple(_PRESETS.keys())


def get_preset(preset_id: str) -> dict[str, Any] | None:
    """A single preset as ``{id, name, description, hardware_spec}`` or None."""
    entry = _PRESETS.get(preset_id)
    if entry is None:
        return None
    return {
        "id": preset_id,
        "name": entry["name"],
        "description": entry["description"],
        "hardware_spec": entry["factory"]().to_dict(),
    }


def list_presets() -> list[dict[str, Any]]:
    """All presets as ``[{id, name, description, hardware_spec}]``."""
    return [get_preset(pid) for pid in PRESET_IDS]  # type: ignore[misc]
