"""Custom-opponent registry endpoints (local files, no auth, no DB).

A "custom opponent" is a user-authored behavior DSL plus a :class:`HardwareSpec`
(recorded for the design; the DSL drives the actual sumo behaviour). These
endpoints let the UI author/validate/save/list/load/delete opponents. A saved
opponent's id can then be passed as ``b_opponent_id`` to the Arena battle
endpoint, or referenced in a training opponent mix.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, status

from webapp.backend import opponents_store
from webapp.shared.opponent_dsl import validate as validate_dsl

router = APIRouter(prefix="/api/opponents", tags=["opponents"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_opponent(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Save a custom opponent. Body: ``{name, hardware_spec, behavior_dsl}``.

    Returns the full record. An invalid spec or DSL yields a 422.
    """
    name = body.get("name")
    hardware_spec = body.get("hardware_spec")
    behavior_dsl = body.get("behavior_dsl")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="name must be a non-empty string",
        )
    try:
        return opponents_store.save_opponent(name, hardware_spec, behavior_dsl)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.post("/validate")
def validate_opponent(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Validate a behavior DSL without saving. Returns ``{ok, errors}``."""
    errors = validate_dsl(body.get("behavior_dsl"))
    return {"ok": not errors, "errors": errors}


@router.get("")
def list_opponents() -> list[dict[str, Any]]:
    """Summaries of all saved custom opponents."""
    return opponents_store.list_opponents()


@router.get("/{opponent_id}")
def get_opponent(opponent_id: str) -> dict[str, Any]:
    """The full record for ``opponent_id``; 404 if missing."""
    record = opponents_store.get_opponent(opponent_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"unknown opponent: {opponent_id}"
        )
    return record


@router.delete("/{opponent_id}")
def delete_opponent(opponent_id: str) -> dict[str, bool]:
    """Delete a custom opponent; ``{deleted: true}`` or 404 if it didn't exist."""
    if not opponents_store.delete_opponent(opponent_id):
        raise HTTPException(
            status_code=404, detail=f"unknown opponent: {opponent_id}"
        )
    return {"deleted": True}
