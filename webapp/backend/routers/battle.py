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
    """Run a battle: model A vs another model / a zoo opponent (single), or A
    vs the whole zoo (gauntlet).

    Body: ``{a_model_id, mode="single"|"gauntlet", b_model_id?, b_opponent_id?,
    rounds=5, mult=3.0, seed=4242, a_spec?, b_spec?, include_held_out?,
    include_custom?}``. For ``single`` exactly one of ``b_model_id`` /
    ``b_opponent_id`` is required and the result is ``{battle_id, mode, stats,
    rounds, trajectory, notes?}``. For ``gauntlet`` side B is ignored and the
    result is ``{battle_id, mode, per_opponent, overall_stats, notes?}``.
    """
    try:
        return battle_mod.run_battle(req)
    except battle_mod.BattleError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.get("/{battle_id}/trajectory")
def get_trajectory(battle_id: str) -> dict[str, Any]:
    """The representative (first decisive) trajectory for a battle, 404 if
    unknown. Back-compat — use ``/trajectory/{ref}`` to fetch a specific
    round."""
    traj = battle_mod.load_trajectory(battle_id)
    if traj is None:
        raise HTTPException(
            status_code=404, detail=f"unknown battle: {battle_id}"
        )
    return traj


@router.get("/{battle_id}/trajectory/{ref}")
def get_trajectory_ref(battle_id: str, ref: str) -> dict[str, Any]:
    """A specific round's recorded trajectory JSON (``<ref>.json``), or 404 if
    unknown / the ref is not a safe filename."""
    traj = battle_mod.load_trajectory_ref(battle_id, ref)
    if traj is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown battle round: {battle_id}/{ref}",
        )
    return traj
