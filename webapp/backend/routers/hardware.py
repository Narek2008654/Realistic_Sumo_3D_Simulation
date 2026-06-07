"""Hardware-spec endpoints: validate a spec and export its geometry.

``POST /api/hardware/validate`` turns a raw :class:`HardwareSpec` dict into
its derived obs/action sizes + signature, checks the generated URDF actually
loads in a throwaway PyBullet DIRECT client, and lists finetune candidates.

``POST /api/hardware/geometry`` returns the three.js-friendly primitive
geometry for the spec.

The request body is the plain dict produced by ``HardwareSpec.to_dict()``.
We accept it as an arbitrary JSON object and let ``HardwareSpec.from_dict``
do the structural validation, surfacing any error as a 422.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from webapp.backend import registry
from webapp.shared.geometry_export import spec_to_geometry
from webapp.shared.hardware_spec import HardwareSpec
from webapp.shared.urdf_gen import generate_urdf

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


def _parse_spec(body: dict[str, Any]) -> HardwareSpec:
    """Build a HardwareSpec from a request body or raise a 422."""
    try:
        return HardwareSpec.from_dict(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid HardwareSpec: {exc}"
        ) from exc


def _urdf_loads_in_pybullet(urdf_str: str) -> tuple[bool, str | None]:
    """Try to load ``urdf_str`` in a headless PyBullet client.

    Returns ``(ok, error)``. Imported lazily so listing geometry never spins
    up a physics server. Any failure (import, connect, parse, load) is caught
    and returned as the error string rather than raised.
    """
    try:
        import pybullet as p
    except Exception as exc:  # pybullet missing / DLL issue
        return False, f"pybullet import failed: {exc}"

    client = -1
    tmp_path: Path | None = None
    try:
        # PyBullet's loadURDF wants a file path, not a string.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(urdf_str)
            tmp_path = Path(fh.name)

        client = p.connect(p.DIRECT)
        if client < 0:
            return False, "failed to connect to PyBullet DIRECT server"
        p.loadURDF(str(tmp_path), physicsClientId=client)
        return True, None
    except Exception as exc:  # noqa: BLE001 - report any load failure
        return False, str(exc)
    finally:
        if client >= 0:
            try:
                p.disconnect(client)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@router.post("/validate")
def validate(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Validate a HardwareSpec: dims, signature, URDF load, candidates."""
    errors: list[str] = []
    spec = _parse_spec(body)

    urdf_valid = False
    try:
        urdf_str = generate_urdf(spec)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"urdf generation failed: {exc}")
    else:
        urdf_valid, load_err = _urdf_loads_in_pybullet(urdf_str)
        if load_err:
            errors.append(load_err)

    signature = spec.obs_signature_hash
    candidates = registry.finetune_candidates(signature, spec.action_dim)

    return {
        "obs_dim": spec.obs_dim,
        "action_dim": spec.action_dim,
        "obs_signature_hash": signature,
        "urdf_valid": urdf_valid,
        "errors": errors,
        "finetune_candidates": candidates,
    }


@router.post("/geometry")
def geometry(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Return the three.js primitive geometry for a HardwareSpec."""
    spec = _parse_spec(body)
    try:
        return spec_to_geometry(spec)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"geometry export failed: {exc}"
        ) from exc
