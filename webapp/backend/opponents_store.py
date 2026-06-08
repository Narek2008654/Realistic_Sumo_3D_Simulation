"""Local-file store for user-authored custom opponents.

No DB, no auth — every saved opponent is a directory under
``data/opponents/<id>/`` holding a single file:

    opponent.json   {id, name, hardware_spec, behavior_dsl, created_at, notes?}

The ``<id>`` is a filesystem-safe slug of the opponent's name plus a short
random suffix, so two opponents saved under the same display name never
collide. All id handling is path-safe (a single sanitised segment), mirroring
:mod:`webapp.backend.robots_store`.

Saving validates BOTH the hardware spec (round-tripped through
:meth:`HardwareSpec.from_dict`) and the behavior DSL (via
:func:`opponent_dsl.validate`) before writing; either failing raises
:class:`ValueError` for the router to translate into a 422.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from webapp.backend import config
from webapp.shared.hardware_spec import HardwareSpec
from webapp.shared.opponent_dsl import OpponentDSL, validate

__all__ = [
    "save_opponent",
    "list_opponents",
    "get_opponent",
    "delete_opponent",
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
    name: str, hardware_spec: dict[str, Any], behavior_dsl: Any
) -> dict[str, Any]:
    """Validate + persist a custom opponent, returning its full record.

    ``hardware_spec`` is a plain ``HardwareSpec.to_dict()`` payload;
    ``behavior_dsl`` is the rule-DSL dict/list. An invalid spec or DSL raises
    :class:`ValueError` (the router maps this to 422).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    if not isinstance(hardware_spec, dict):
        raise ValueError("hardware_spec must be an object")
    try:
        spec = HardwareSpec.from_dict(hardware_spec)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid HardwareSpec: {exc}") from exc

    dsl_errors = validate(behavior_dsl)
    if dsl_errors:
        raise ValueError("invalid behavior_dsl: " + "; ".join(dsl_errors))
    # Canonicalise the DSL ({rules, default}) so the stored form is stable.
    canon_dsl = OpponentDSL.from_dict(behavior_dsl).to_dict()

    opp_id = _new_id(name)
    record = {
        "id": opp_id,
        "name": name.strip(),
        "hardware_spec": spec.to_dict(),
        "behavior_dsl": canon_dsl,
        "created_at": _iso_now(),
        "notes": _STANDARD_CHASSIS_NOTE,
    }

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
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "name": record["name"],
        "created_at": record["created_at"],
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
