"""ARENA battle endpoints.

``POST /api/battle`` runs a synchronous head-to-head battle (short) in-process
under the PyBullet lock and returns aggregated stats + a recorded trajectory.
``GET /api/battle/{id}/trajectory`` re-reads a recorded trajectory.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from webapp.backend import battle as battle_mod

router = APIRouter(prefix="/api/battle", tags=["battle"])


@router.post("")
def run_battle(req: dict[str, Any]) -> dict[str, Any]:
    """Run a battle: model A vs another model OR a zoo opponent.

    Body: ``{a_model_id, b_model_id?, b_opponent_id?, rounds=5, mult=3.0,
    seed=4242, a_spec?, b_spec?}``. Exactly one of ``b_model_id`` /
    ``b_opponent_id`` is required. Returns
    ``{battle_id, stats, trajectory, notes?}``.
    """
    try:
        return battle_mod.run_battle(req)
    except battle_mod.BattleError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.get("/{battle_id}/trajectory")
def get_trajectory(battle_id: str) -> dict[str, Any]:
    """The recorded trajectory JSON for a battle, or 404 if unknown."""
    traj = battle_mod.load_trajectory(battle_id)
    if traj is None:
        raise HTTPException(
            status_code=404, detail=f"unknown battle: {battle_id}"
        )
    return traj
