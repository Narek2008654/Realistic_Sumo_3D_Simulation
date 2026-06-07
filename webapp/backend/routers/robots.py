"""Saved-robot registry endpoints (local files, no auth, no DB).

A "saved robot" is a designed :class:`HardwareSpec` plus its generated URDF,
persisted under ``data/robots/<id>/`` by :mod:`webapp.backend.robots_store`.
These endpoints let the UI save a design, list/load saved designs, fetch the
URDF, and delete a design.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import PlainTextResponse

from webapp.backend import robots_store

router = APIRouter(prefix="/api/robots", tags=["robots"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_robot(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Save a designed robot. Body: ``{name: str, hardware_spec: dict}``.

    Returns the full record (id, metadata, hardware_spec). An invalid spec
    (fails ``HardwareSpec.from_dict`` or URDF generation) yields a 400.
    """
    name = body.get("name")
    spec_dict = body.get("hardware_spec")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="name must be a non-empty string")
    if not isinstance(spec_dict, dict):
        raise HTTPException(
            status_code=400, detail="hardware_spec must be an object"
        )
    try:
        return robots_store.save_robot(name, spec_dict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def list_robots() -> list[dict[str, Any]]:
    """Summaries of all saved robots (no full hardware_spec)."""
    return robots_store.list_robots()


@router.get("/{robot_id}")
def get_robot(robot_id: str) -> dict[str, Any]:
    """The full record (incl hardware_spec) for ``robot_id``; 404 if missing."""
    record = robots_store.get_robot(robot_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown robot: {robot_id}")
    return record


@router.get("/{robot_id}/urdf", response_class=PlainTextResponse)
def get_robot_urdf(robot_id: str) -> PlainTextResponse:
    """The generated URDF text for ``robot_id`` (text/plain); 404 if missing."""
    urdf = robots_store.get_urdf(robot_id)
    if urdf is None:
        raise HTTPException(status_code=404, detail=f"unknown robot: {robot_id}")
    return PlainTextResponse(urdf, media_type="text/plain")


@router.delete("/{robot_id}")
def delete_robot(robot_id: str) -> dict[str, bool]:
    """Delete a saved robot; ``{deleted: true}`` or 404 if it didn't exist."""
    if not robots_store.delete_robot(robot_id):
        raise HTTPException(status_code=404, detail=f"unknown robot: {robot_id}")
    return {"deleted": True}
