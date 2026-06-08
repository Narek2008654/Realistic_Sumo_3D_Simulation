"""Custom-opponent registry endpoints (local files, no auth, no DB).

A "custom opponent" is BEHAVIOR x HARDWARE: a :class:`HardwareSpec` (a custom
chassis or a hardware preset) crossed with a behavior that is EITHER a built-in
zoo controller (``{"kind":"zoo","zoo_id":...}``) OR a user-authored rule DSL
(``{"kind":"dsl","dsl":...}``). This lets a user save variants like "Heavy
Dodger" or "Fast Novamax" as well as fully custom-DSL opponents. These
endpoints let the UI author/validate/save/list/load/delete opponents. A saved
opponent's id can then be passed as ``b_opponent_id`` to the Arena battle
endpoint, or referenced in a training opponent mix.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, status

from webapp.backend import opponents_store

router = APIRouter(prefix="/api/opponents", tags=["opponents"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_opponent(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Save a custom opponent. Body: ``{name, hardware_spec, behavior}`` where
    ``behavior`` is ``{kind:"zoo",zoo_id} | {kind:"dsl",dsl}``.

    For back-compat a legacy ``behavior_dsl`` (the rule DSL directly) is still
    accepted in place of ``behavior``. Returns the full record. An invalid spec
    or behavior yields a 422.
    """
    name = body.get("name")
    hardware_spec = body.get("hardware_spec")
    behavior = body.get("behavior")
    behavior_dsl = body.get("behavior_dsl")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="name must be a non-empty string",
        )
    try:
        return opponents_store.save_opponent(
            name, hardware_spec, behavior=behavior, behavior_dsl=behavior_dsl
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.post("/validate")
def validate_opponent(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Validate a behavior without saving. Returns ``{ok, errors}``.

    Accepts the new ``behavior`` object (zoo|dsl) or a legacy ``behavior_dsl``.
    """
    behavior = body.get("behavior")
    if behavior is None and "behavior_dsl" in body:
        behavior = {"kind": "dsl", "dsl": body["behavior_dsl"]}
    errors = opponents_store.validate_behavior(behavior)
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
