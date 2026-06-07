"""LITE FastAPI app: no auth, no DB, local files only.

Wires the hardware / models / assets routers, a permissive dev CORS policy
for ``http://localhost:*`` (and ``127.0.0.1``), and a health probe. Run with
``webapp/run_local.bat`` or::

    python -m uvicorn webapp.backend.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend import config
from webapp.backend.routers import assets, hardware, models

app = FastAPI(
    title="Realistic Sumo 3D — LITE backend",
    description="Local, auth-less, DB-less API for the sumo web UI.",
    version="0.1.0",
)

# Dev CORS: any localhost / 127.0.0.1 port. Regex covers the changing Vite
# port. This is a local-only LITE app — no credentials, no auth.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(hardware.router)
app.include_router(models.router)
app.include_router(assets.router)


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()


@app.get("/api/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
