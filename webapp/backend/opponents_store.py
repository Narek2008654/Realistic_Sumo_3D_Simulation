"""Local-file store for user-authored custom opponents.

No DB, no auth — every saved opponent is a directory under
``data/opponents/<id>/`` holding a single file:

    opponent.json   {id, name, hardware_spec, behavior, behavior_dsl?,
                     created_at, notes?}

A saved opponent is BEHAVIOR x HARDWARE: any chassis (a custom HardwareSpec or
a hardware preset) crossed with any behavior. ``behavior`` is one of two kinds:

    {"kind": "zoo", "zoo_id": "dodger"}   # a built-in zoo controller
    {"kind": "dsl", "dsl": {rules, default}}  # a user-authored rule DSL

This lets a user save variants like "Heavy Dodger" (the built-in dodger
behavior on a heavy chassis) or "Fast Novamax" (the novamax behavior on a
light/fast chassis) alongside fully custom-DSL opponents.

**Back-compat:** a legacy record carrying only ``behavior_dsl`` (and no
``behavior``) is normalised on read to ``{"kind":"dsl","dsl": behavior_dsl}``.
New DSL records keep writing ``behavior_dsl`` too (a cheap mirror) so older
readers — the trainers' run-config bootstrap — keep working unchanged.

The ``<id>`` is a filesystem-safe slug of the opponent's name plus a short
random suffix, so two opponents saved under the same display name never
collide. All id handling is path-safe (a single sanitised segment), mirroring
:mod:`webapp.backend.robots_store`.

Saving validates the hardware spec (round-tripped through
:meth:`HardwareSpec.from_dict`) and the behavior (a ``zoo`` kind must name a
``zoo_id`` in :data:`opponents.OPPONENT_REGISTRY`; a ``dsl`` kind is checked
via :func:`opponent_dsl.validate`); either failing raises :class:`ValueError`
for the router to translate into a 422.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from webapp.backend import config
from webapp.shared.hardware_spec import HardwareSpec, mini_sumo_violations
from webapp.shared.opponent_dsl import OpponentDSL, validate

__all__ = [
    "save_opponent",
    "list_opponents",
    "get_opponent",
    "delete_opponent",
    "normalize_behavior",
    "validate_behavior",
    "behavior_summary",
    "build_controller",
]

_JSON_NAME = "opponent.json"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# The custom opponent fights on ITS OWN hardware: at battle time the enemy
# body is generated from this saved hardware_spec (chassis/wheels/wedge/mass)
# and its motor caps come from spec.drivetrain, while the behavior_dsl drives
# the controller. (The opponent's sensor/perception model still uses the
# standard 5-key IR + line-sensor suite the DSL predicates map to.)
_STANDARD_CHASSIS_NOTE = (
    "Custom opponent fights on its own hardware: the enemy chassis, wheels, "
    "wedge, mass and motor caps come from its saved hardware_spec at battle "
    "time. The behavior_dsl drives this opponent's controller; its perception "
    "uses the standard IR + line-sensor suite."
)


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Behavior: zoo OR dsl. ONE place decides which — every consumer (battle,
# training-config resolution, the trainers) builds its controller through
# ``build_controller`` so the zoo-vs-dsl branch never gets duplicated.
# ---------------------------------------------------------------------------
def normalize_behavior(record_or_behavior: Any) -> dict[str, Any]:
    """Return the canonical ``behavior`` object for a record (or a raw behavior).

    Accepts either a full opponent record or a bare behavior dict. A record
    that already carries ``behavior`` is returned as-is; a legacy record with
    only ``behavior_dsl`` is migrated on read to ``{"kind":"dsl","dsl": ...}``.
    Raises :class:`ValueError` if no behavior can be derived.
    """
    if isinstance(record_or_behavior, dict) and "kind" in record_or_behavior:
        return record_or_behavior  # already a behavior object
    if isinstance(record_or_behavior, dict):
        if isinstance(record_or_behavior.get("behavior"), dict):
            return record_or_behavior["behavior"]
        if "behavior_dsl" in record_or_behavior:  # legacy migrate-on-read
            return {"kind": "dsl", "dsl": record_or_behavior["behavior_dsl"]}
    raise ValueError("record has no behavior (neither 'behavior' nor 'behavior_dsl')")


def validate_behavior(behavior: Any) -> list[str]:
    """Validate a ``behavior`` object, returning human-readable errors (empty
    == valid). A ``zoo`` behavior must name a known zoo id; a ``dsl`` behavior
    is validated via :func:`opponent_dsl.validate`."""
    if not isinstance(behavior, dict) or "kind" not in behavior:
        return ["behavior must be an object with a 'kind' of 'zoo' or 'dsl'"]
    kind = behavior.get("kind")
    if kind == "zoo":
        zoo_id = behavior.get("zoo_id")
        # Lazy import: the zoo registry pulls sumo_env.
        from opponents import OPPONENT_REGISTRY

        if not isinstance(zoo_id, str) or zoo_id not in OPPONENT_REGISTRY:
            return [
                f"behavior.zoo_id must be one of {sorted(OPPONENT_REGISTRY)}, "
                f"got {zoo_id!r}"
            ]
        return []
    if kind == "dsl":
        return validate(behavior.get("dsl"))
    return [f"behavior.kind must be 'zoo' or 'dsl', got {kind!r}"]


def _canon_behavior(behavior: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise a (validated) behavior for stable on-disk storage."""
    if behavior["kind"] == "zoo":
        return {"kind": "zoo", "zoo_id": behavior["zoo_id"]}
    return {"kind": "dsl", "dsl": OpponentDSL.from_dict(behavior["dsl"]).to_dict()}


def behavior_summary(behavior: dict[str, Any]) -> str:
    """A short human label for the UI: ``"zoo:dodger"`` / ``"custom rules"``."""
    if behavior.get("kind") == "zoo":
        return f"zoo:{behavior.get('zoo_id')}"
    return "custom rules"


def build_controller(record_or_behavior: Any):
    """The ONE shared factory: build an ``OpponentController`` from a saved
    opponent record (or a bare behavior object).

    * ``zoo`` -> ``OPPONENT_REGISTRY[zoo_id]()`` (a fresh built-in controller).
    * ``dsl`` -> ``DslOpponent(OpponentDSL.from_dict(dsl))`` (pure interpreter,
      no eval/exec).

    Reused by battle, training-config resolution, and the trainers so the
    zoo-vs-dsl decision lives in exactly one place — the pure
    :func:`opponents.build_controller_from_behavior` helper this delegates to.
    """
    behavior = normalize_behavior(record_or_behavior)
    from opponents import build_controller_from_behavior

    return build_controller_from_behavior(behavior)


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "opponent"


def _is_safe_id(opp_id: str) -> bool:
    if not isinstance(opp_id, str) or not opp_id:
        return False
    if "/" in opp_id or "\\" in opp_id or ".." in opp_id:
        return False
    return bool(_VALID_ID_RE.match(opp_id))


def _new_id(name: str) -> str:
    base = _slugify(name)
    config.ensure_dirs()
    for _ in range(1000):
        candidate = f"{base}-{secrets.token_hex(3)}"
        if not (config.OPPONENTS_DIR / candidate).exists():
            return candidate
    raise RuntimeError("could not allocate a unique opponent id")  # pragma: no cover


def save_opponent(
    name: str,
    hardware_spec: dict[str, Any],
    behavior: Any = None,
    behavior_dsl: Any = None,
) -> dict[str, Any]:
    """Validate + persist a custom opponent (BEHAVIOR x HARDWARE), returning
    its full record.

    ``hardware_spec`` is a plain ``HardwareSpec.to_dict()`` payload.
    ``behavior`` is the new ``{kind:"zoo"|"dsl", ...}`` object. For back-compat
    a caller may instead pass ``behavior_dsl`` (the rule DSL directly), which
    is treated as ``{"kind":"dsl","dsl": behavior_dsl}``. Exactly one of the two
    must be supplied. An invalid spec or behavior raises :class:`ValueError`
    (the router maps this to 422).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    if not isinstance(hardware_spec, dict):
        raise ValueError("hardware_spec must be an object")
    try:
        spec = HardwareSpec.from_dict(hardware_spec)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid HardwareSpec: {exc}") from exc
    violations = mini_sumo_violations(spec)
    if violations:
        raise ValueError("not mini-sumo legal: " + "; ".join(violations))

    # Resolve the behavior from either the new ``behavior`` object or the legacy
    # ``behavior_dsl`` shorthand (exactly one).
    if behavior is None and behavior_dsl is not None:
        behavior = {"kind": "dsl", "dsl": behavior_dsl}
    if behavior is None:
        raise ValueError("a behavior (or behavior_dsl) is required")

    errs = validate_behavior(behavior)
    if errs:
        raise ValueError("invalid behavior: " + "; ".join(errs))
    canon_behavior = _canon_behavior(behavior)

    opp_id = _new_id(name)
    record = {
        "id": opp_id,
        "name": name.strip(),
        "hardware_spec": spec.to_dict(),
        "behavior": canon_behavior,
        "created_at": _iso_now(),
        "notes": _STANDARD_CHASSIS_NOTE,
    }
    # Cheap mirror: keep writing ``behavior_dsl`` for DSL behaviors so older
    # readers (the trainers' run-config bootstrap) keep working unchanged.
    if canon_behavior["kind"] == "dsl":
        record["behavior_dsl"] = canon_behavior["dsl"]

    opp_dir = config.OPPONENTS_DIR / opp_id
    opp_dir.mkdir(parents=True, exist_ok=True)
    (opp_dir / _JSON_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def _read_record(opp_id: str) -> dict[str, Any] | None:
    if not _is_safe_id(opp_id):
        return None
    path = config.OPPONENTS_DIR / opp_id / _JSON_NAME
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    # Migrate-on-read: a legacy record (behavior_dsl only) gets a synthesised
    # ``behavior`` so every consumer sees the new shape. Not persisted here.
    if isinstance(record, dict) and "behavior" not in record:
        try:
            record["behavior"] = normalize_behavior(record)
        except ValueError:
            pass
    return record


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    behavior = record.get("behavior")
    if behavior is None:
        try:
            behavior = normalize_behavior(record)
        except ValueError:
            behavior = None
    return {
        "id": record["id"],
        "name": record["name"],
        "created_at": record["created_at"],
        "behavior": behavior,
        "behavior_summary": behavior_summary(behavior) if behavior else None,
    }


def list_opponents() -> list[dict[str, Any]]:
    """Summaries of every saved custom opponent, newest first."""
    config.ensure_dirs()
    summaries: list[dict[str, Any]] = []
    for child in config.OPPONENTS_DIR.iterdir():
        if not child.is_dir():
            continue
        record = _read_record(child.name)
        if record is not None:
            summaries.append(_summary(record))
    summaries.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return summaries


def get_opponent(opp_id: str) -> dict[str, Any] | None:
    """The full record (incl hardware_spec + behavior_dsl), or None."""
    return _read_record(opp_id)


def delete_opponent(opp_id: str) -> bool:
    """Delete a saved opponent's directory; True if it existed, else False."""
    if not _is_safe_id(opp_id):
        return False
    opp_dir = config.OPPONENTS_DIR / opp_id
    if not opp_dir.is_dir():
        return False
    for child in opp_dir.iterdir():
        try:
            child.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
    opp_dir.rmdir()
    return True
