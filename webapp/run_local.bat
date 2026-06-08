@echo off
REM Launch the LITE backend locally (no auth, no DB).
REM Activates the sumo conda env so torch/pybullet DLLs resolve, then serves
REM the FastAPI app on 127.0.0.1:8000.
REM
REM NOTE: --reload is intentionally OFF. With autoreload, editing any backend
REM file restarts the worker mid-run; that used to drop the live training-job
REM handle (job tracking now re-adopts running jobs from disk, but a reload
REM still blips in-flight requests). Restart this script manually when you want
REM backend code changes to take effect.

call C:\Users\User\miniforge3\Scripts\activate.bat sumo
set KMP_DUPLICATE_LIB_OK=TRUE

REM Run from the repo root (one level up from this script's dir).
pushd "%~dp0.."
python -m uvicorn webapp.backend.app:app --host 127.0.0.1 --port 8000
popd
