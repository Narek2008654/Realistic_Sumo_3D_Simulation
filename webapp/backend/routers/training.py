"""Training-orchestration endpoints (LITE: single local job, no auth/DB).

Launch one training run, stream its progress, stop it, list past runs, and
get a recommended config for a hardware spec. The heavy lifting lives in
:mod:`webapp.backend.training` (job manager) and
:mod:`webapp.backend.services.recommender` (config heuristics).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from webapp.backend import training
from webapp.backend.services import recommender
from webapp.shared.hardware_spec import HardwareSpec

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("")
def start_training(body: dict[str, Any] = Body(...)) -> dict[str, str]:
    """Start a training job. Returns ``{job_id}``; 409 if one is running.

    Body: ``{robot_id?|hardware_spec?, algo, mode, base_model_id?,
    total_steps?, eval_every?, opponent_weights?, smoke?}``.
    """
    try:
        job_id = training.start_job(body)
    except training.JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except training.JobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id}


@router.get("/status")
def status_active() -> dict[str, Any]:
    """Status of the currently-active job (or an ``idle`` payload)."""
    return training.status(None)


@router.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    """Summaries of all jobs on disk (newest first)."""
    return training.list_jobs()


@router.post("/stop")
def stop_training() -> dict[str, bool]:
    """Stop the active job; ``{stopped: true}`` if one was running."""
    return {"stopped": training.stop()}


@router.post("/recommend")
def recommend(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Recommend a training config. Body: ``{hardware_spec?, mode}``.

    ``hardware_spec`` defaults to the canonical robot when omitted.
    """
    spec_dict = body.get("hardware_spec")
    if spec_dict is None:
        spec = HardwareSpec.default()
    else:
        if not isinstance(spec_dict, dict):
            raise HTTPException(
                status_code=400, detail="hardware_spec must be an object"
            )
        try:
            spec = HardwareSpec.from_dict(spec_dict)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    mode = str(body.get("mode", "scratch"))
    return recommender.recommend(spec, mode)


@router.get("/status/{job_id}")
def status_by_id(job_id: str) -> dict[str, Any]:
    """Status of a specific job by id; ``unknown`` state if not on disk."""
    return training.status(job_id)
