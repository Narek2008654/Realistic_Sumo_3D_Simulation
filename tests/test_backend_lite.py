"""Smoke tests for the LITE backend.

Runnable two ways:
    python tests/test_backend_lite.py     # plain-python harness (prints PASS/FAIL)
    pytest tests/test_backend_lite.py      # standard pytest

Prefers FastAPI's ``TestClient`` when ``httpx`` is installed; otherwise falls
back to a tiny in-process ASGI client built on ``anyio`` (a FastAPI dep), so
the suite runs even in the lean ``sumo`` env that has no httpx — no new pip
deps required.

The slow ``/api/models/{id}/evaluate`` endpoint is intentionally NOT exercised
here — it runs real physics rollouts.
"""

from __future__ import annotations

import importlib.util
import json as _json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow running as a bare script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.backend.app import app  # noqa: E402
from webapp.shared.hardware_spec import HardwareSpec  # noqa: E402


class _Resp:
    """Minimal response shim (status_code / json() / text)."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status_code = status
        self._body = body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", "replace")

    def json(self) -> Any:
        return _json.loads(self._body)


class _AnyioClient:
    """Drive the ASGI app in-process via anyio (no httpx needed)."""

    def __init__(self, asgi_app: Any) -> None:
        self._app = asgi_app

    def _call(self, method: str, path: str, body: Any | None) -> _Resp:
        import anyio

        payload = b"" if body is None else _json.dumps(body).encode("utf-8")
        raw_path, _, query = path.partition("?")
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": raw_path,
            "raw_path": raw_path.encode("utf-8"),
            "query_string": query.encode("utf-8"),
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        sent = {"body": b"", "status": 500}

        async def run() -> None:
            req_done = False

            async def receive() -> dict[str, Any]:
                nonlocal req_done
                if not req_done:
                    req_done = True
                    return {"type": "http.request", "body": payload,
                            "more_body": False}
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    sent["status"] = message["status"]
                elif message["type"] == "http.response.body":
                    sent["body"] += message.get("body", b"")

            await self._app(scope, receive, send)

        anyio.run(run)
        return _Resp(sent["status"], sent["body"])

    def get(self, path: str) -> _Resp:
        return self._call("GET", path, None)

    def post(self, path: str, json: Any | None = None) -> _Resp:
        return self._call("POST", path, json)


def _make_client() -> Any:
    if importlib.util.find_spec("httpx") is not None:
        from fastapi.testclient import TestClient

        return TestClient(app)
    return _AnyioClient(app)


client = _make_client()


def test_health() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ok"}


def test_hardware_validate_default() -> None:
    spec = HardwareSpec.default()
    resp = client.post("/api/hardware/validate", json=spec.to_dict())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["obs_dim"] == 21, body
    assert body["action_dim"] == 9, body
    assert body["urdf_valid"] is True, body
    assert body["obs_signature_hash"], body
    assert isinstance(body["finetune_candidates"], list), body


def test_models_list_contains_committed() -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200, resp.text
    cards = resp.json()
    assert isinstance(cards, list) and cards, cards
    by_id = {c["id"]: c for c in cards}
    assert "ppo_robust_best" in by_id, sorted(by_id)
    ppo = by_id["ppo_robust_best"]
    assert ppo["obs_dim"] == 21, ppo
    assert ppo["action_dim"] == 9, ppo
    assert ppo["algo"] == "ppo", ppo


def test_finetune_candidates_match_signature() -> None:
    # ppo_robust_best uses the default 21/9 contract.
    resp = client.get("/api/models/ppo_robust_best/finetune-candidates")
    assert resp.status_code == 200, resp.text
    cands = resp.json()
    assert isinstance(cands, list) and cands, cands
    default_sig = HardwareSpec.default().obs_signature_hash
    for c in cands:
        assert c["obs_signature_hash"] == default_sig, c
        assert c["obs_dim"] == 21, c
        assert c["action_dim"] == 9, c


def test_hardware_geometry_default_has_wheels() -> None:
    spec = HardwareSpec.default()
    resp = client.post("/api/hardware/geometry", json=spec.to_dict())
    assert resp.status_code == 200, resp.text
    geom = resp.json()
    assert "links" in geom and "joints" in geom, geom
    link_names = {l["name"] for l in geom["links"]}
    assert "left_wheel" in link_names, link_names
    assert "right_wheel" in link_names, link_names


# ---------------------------------------------------------------------------
# Plain-python harness (no pytest required).
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        ("health", test_health),
        ("hardware_validate_default", test_hardware_validate_default),
        ("models_list_contains_committed", test_models_list_contains_committed),
        ("finetune_candidates_match_signature",
         test_finetune_candidates_match_signature),
        ("hardware_geometry_default_has_wheels",
         test_hardware_geometry_default_has_wheels),
    ]
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
            print(f"  FAIL  {name}: {exc!r}")
    print(f"\nPASS: {len(passed)} / FAIL: {len(failed)}")
    for name, err in failed:
        print(f"    - {name}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_main())
