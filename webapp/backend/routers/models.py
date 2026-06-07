"""Model-registry endpoints over the committed checkpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from webapp.backend import registry

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models() -> list[dict[str, Any]]:
    """All registry cards (one per recognised ``checkpoints/*.pt``)."""
    return registry.list_models()


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


@router.post("/{model_id}/evaluate")
def evaluate(model_id: str) -> dict[str, Any]:
    """Run headless rollouts and cache the metrics (slow; on demand only)."""
    card = registry.evaluate_model(model_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return card
