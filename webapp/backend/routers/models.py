"""Model-registry endpoints over the committed checkpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from webapp.backend import registry

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models() -> list[dict[str, Any]]:
    """All registry cards (one per recognised ``checkpoints/*.pt``)."""
    return registry.list_models()


# NOTE: the static ``/runs`` and ``/promote`` routes MUST be declared before
# the dynamic ``/{model_id}`` route below — otherwise FastAPI matches them as a
# model id and they 404.
@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    """One summary per training-job dir (source for the promote picker)."""
    return registry.list_runs()


@router.post("/promote")
def promote(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Promote a finished job's ``best``/``final`` checkpoint into the registry.

    Body: ``{job_id, which: 'best'|'final', name}``. Returns the new ModelCard.
    404 unknown job / missing ``<which>.pt``; 409 name collision or protected
    slug; 422 bad ``which`` or empty/unsafe ``name``.
    """
    try:
        return registry.promote_job_model(
            job_id=str(payload.get("job_id") or ""),
            which=str(payload.get("which") or "best"),
            name=payload.get("name"),
        )
    except registry.PromoteNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except registry.PromoteConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except registry.PromoteInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{model_id}")
def get_model(model_id: str) -> dict[str, Any]:
    """One model card by id (filename stem)."""
    card = registry.get_model(model_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return card


@router.get("/{model_id}/finetune-candidates")
def finetune_candidates(model_id: str) -> list[dict[str, Any]]:
    """Cards byte-compatible with ``model_id``'s obs/action contract."""
    card = registry.get_model(model_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return registry.finetune_candidates(
        card.get("obs_signature_hash"), card.get("action_dim")
    )


@router.delete("/{model_id}")
def delete_model(model_id: str) -> dict[str, Any]:
    """Delete a checkpoint + its cached card. 404 unknown, 409 if protected."""
    try:
        deleted = registry.delete_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return {"deleted": True, "id": model_id}


@router.post("/{model_id}/evaluate")
def evaluate(model_id: str, mode: str = "quick") -> dict[str, Any]:
    """Run headless rollouts and cache the metrics (slow; on demand only).

    ``mode`` = ``quick`` (3-opponent probe) or ``full`` (whole zoo + held-out).
    """
    if mode not in ("quick", "full"):
        raise HTTPException(
            status_code=422, detail="mode must be 'quick' or 'full'"
        )
    card = registry.evaluate_model(model_id, mode=mode)
    if card is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return card
