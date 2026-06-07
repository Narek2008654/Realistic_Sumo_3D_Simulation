"""Local-file store for user-designed robots (HardwareSpec + URDF).

No DB, no auth — every saved robot is a directory under
``data/robots/<id>/`` holding two files:

    robot.json   metadata + the full ``HardwareSpec.to_dict()`` payload
    robot.urdf   the generated URDF XML for that spec

The ``<id>`` is a filesystem-safe slug of the robot's name plus a short
random suffix, so two robots saved under the same display name never collide
or overwrite each other. All id handling is path-safe: an id is only ever a
single, sanitised path segment, so a caller-supplied id can never traverse
out of ``ROBOTS_DIR``.

Saving validates the incoming spec by round-tripping it through
:meth:`HardwareSpec.from_dict` and generating the URDF; an invalid spec is
surfaced as :class:`ValueError` for the router to translate into a 400.
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
from webapp.shared.urdf_gen import generate_urdf

__all__ = [
    "save_robot",
    "list_robots",
    "get_robot",
    "get_urdf",
    "delete_robot",
]

_JSON_NAME = "robot.json"
_URDF_NAME = "robot.urdf"

# A stored id is a slug: lowercase alnum + dashes, with a short hex suffix.
# This pattern is what we accept back from callers (path-safety gate).
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _iso_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _slugify(name: str) -> str:
    """Turn a display name into a filesystem-safe slug base."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "robot"


def _is_safe_id(robot_id: str) -> bool:
    """True iff ``robot_id`` is a single, traversal-free path segment."""
    if not isinstance(robot_id, str) or not robot_id:
        return False
    if "/" in robot_id or "\\" in robot_id or ".." in robot_id:
        return False
    return bool(_VALID_ID_RE.match(robot_id))


def _robot_dir(robot_id: str) -> Path:
    """The directory for ``robot_id`` (validated path-safe first)."""
    if not _is_safe_id(robot_id):
        raise ValueError(f"invalid robot id: {robot_id!r}")
    return config.ROBOTS_DIR / robot_id


def _new_id(name: str) -> str:
    """A unique, path-safe id for ``name`` (slug + short suffix)."""
    base = _slugify(name)
    config.ensure_dirs()
    # Loop until we land on a directory that doesn't already exist. With 6 hex
    # chars of entropy a single retry is already astronomically unlikely.
    for _ in range(1000):
        candidate = f"{base}-{secrets.token_hex(3)}"
        if not (config.ROBOTS_DIR / candidate).exists():
            return candidate
    raise RuntimeError("could not allocate a unique robot id")  # pragma: no cover


def _build_record(
    robot_id: str, name: str, created_at: str, spec: HardwareSpec
) -> dict[str, Any]:
    """Assemble the full on-disk record for a robot."""
    return {
        "id": robot_id,
        "name": name,
        "created_at": created_at,
        "obs_dim": spec.obs_dim,
        "action_dim": spec.action_dim,
        "obs_signature_hash": spec.obs_signature_hash,
        "hardware_spec": spec.to_dict(),
    }


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    """A list view of a record (no full hardware_spec payload)."""
    return {
        "id": record["id"],
        "name": record["name"],
        "created_at": record["created_at"],
        "obs_dim": record["obs_dim"],
        "action_dim": record["action_dim"],
        "obs_signature_hash": record["obs_signature_hash"],
    }


def save_robot(name: str, spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate + persist a robot design, returning its full record.

    ``spec_dict`` is a plain ``HardwareSpec.to_dict()`` payload. It is parsed
    via :meth:`HardwareSpec.from_dict` and its URDF is generated; either
    failing raises :class:`ValueError` (the router maps this to a 400). The
    robot's display name comes from ``name`` (the spec's own ``name`` field is
    left untouched).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    try:
        spec = HardwareSpec.from_dict(spec_dict)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid HardwareSpec: {exc}") from exc

    try:
        urdf = generate_urdf(spec)
    except Exception as exc:  # noqa: BLE001 - any gen failure is a bad spec
        raise ValueError(f"urdf generation failed: {exc}") from exc

    robot_id = _new_id(name)
    record = _build_record(robot_id, name.strip(), _iso_now(), spec)

    robot_dir = _robot_dir(robot_id)
    robot_dir.mkdir(parents=True, exist_ok=True)
    (robot_dir / _JSON_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    (robot_dir / _URDF_NAME).write_text(urdf, encoding="utf-8")
    return record


def _read_record(robot_id: str) -> dict[str, Any] | None:
    """Load a robot's full record, or ``None`` if it doesn't exist."""
    if not _is_safe_id(robot_id):
        return None
    path = config.ROBOTS_DIR / robot_id / _JSON_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_robots() -> list[dict[str, Any]]:
    """Summaries of every saved robot, newest first (by created_at)."""
    config.ensure_dirs()
    summaries: list[dict[str, Any]] = []
    for child in config.ROBOTS_DIR.iterdir():
        if not child.is_dir():
            continue
        record = _read_record(child.name)
        if record is not None:
            summaries.append(_summary(record))
    summaries.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return summaries


def get_robot(robot_id: str) -> dict[str, Any] | None:
    """The full record (incl ``hardware_spec``) for ``robot_id``, or None."""
    return _read_record(robot_id)


def get_urdf(robot_id: str) -> str | None:
    """The generated URDF text for ``robot_id``, or ``None`` if missing."""
    if not _is_safe_id(robot_id):
        return None
    path = config.ROBOTS_DIR / robot_id / _URDF_NAME
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def delete_robot(robot_id: str) -> bool:
    """Delete a saved robot's directory; True if it existed, else False."""
    if not _is_safe_id(robot_id):
        return False
    robot_dir = config.ROBOTS_DIR / robot_id
    if not robot_dir.is_dir():
        return False
    # Remove the two known files then the directory; tolerate stray contents.
    for child in robot_dir.iterdir():
        try:
            child.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
    robot_dir.rmdir()
    return True
