"""Static job-artifact endpoints (trajectories) served from local files."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from webapp.backend import config

router = APIRouter(prefix="/api/trajectories", tags=["assets"])


@router.get("/{job}/{step}")
def get_trajectory(job: str, step: str) -> FileResponse:
    """Serve ``data/jobs/<job>/trajectories/<step>.json`` if it exists.

    ``job`` and ``step`` are validated to be plain path segments (no
    traversal) before touching the filesystem; a missing file yields 404.
    """
    if "/" in job or "\\" in job or ".." in job or \
            "/" in step or "\\" in step or ".." in step:
        raise HTTPException(status_code=404, detail="not found")

    path = config.JOBS_DIR / job / "trajectories" / f"{step}.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"no trajectory for job={job} step={step}"
        )
    return FileResponse(str(path), media_type="application/json")
