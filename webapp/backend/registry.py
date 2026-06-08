"""Local-file model registry over ``checkpoints/*.pt``.

Each committed checkpoint is a bare PyTorch ``state_dict`` of the shared
dueling architecture (``trunk.0/2`` + ``advantage_head`` + ``value_head``).
The checkpoints predate :class:`HardwareSpec`, so we infer everything we can
straight from the weight shapes:

    trunk.0.weight        -> (hidden1, obs_dim)
    trunk.2.weight        -> (hidden2, hidden1)
    advantage_head.weight -> (action_dim, hidden2)

A small JSON "card" is built per checkpoint and cached under
``data/registry/<id>.json``; it is recomputed whenever the ``.pt`` is newer
than its card. Evaluation metrics are filled in on demand by
:func:`evaluate_model` (slow — it runs real rollouts).

Windows DLL-order convention: import ``torch`` before ``numpy``.
"""

from __future__ import annotations

import torch  # noqa: F401  (must precede numpy for Windows DLL ordering)

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from webapp.backend import config
from webapp.backend.pybullet_lock import pybullet_lock
from webapp.shared.hardware_spec import HardwareSpec

__all__ = [
    "scan_checkpoints",
    "list_models",
    "get_model",
    "finetune_candidates",
    "evaluate_model",
    "delete_model",
    "PROTECTED_MODEL_IDS",
]

# Deploy / canonical checkpoints that the UI must NOT delete (these are the
# committed, git-tracked models — the deployed champion + the BC bases). They
# can still be removed deliberately with git if truly needed.
PROTECTED_MODEL_IDS = frozenset({
    "ppo_robust_best",
    "dqn3d_stack_stageA_best",
    "dqn_3d_bc_actor_best",
    "dqn_3d_bc_actor_final",
    "dqn_3d_bc_oldphys_best",
})

# The obs/action contract of TODAY's robot. Legacy checkpoints that happen to
# match it (21-dim obs, 9 actions) are tagged with this signature so the UI
# can offer them as finetune candidates; everything else gets ``null``.
_DEFAULT_SPEC = HardwareSpec.default()
_DEFAULT_SIGNATURE = _DEFAULT_SPEC.obs_signature_hash
_DEFAULT_OBS_DIM = _DEFAULT_SPEC.obs_dim      # 21
_DEFAULT_ACTION_DIM = _DEFAULT_SPEC.action_dim  # 9


def _iso(ts: float) -> str:
    """UTC ISO-8601 timestamp from a POSIX mtime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _infer_arch(pt_path: Path) -> dict[str, Any]:
    """Read a bare state_dict and derive obs/action dims, arch, params.

    Raises ``ValueError`` if the checkpoint isn't the expected dueling shape.
    """
    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    try:
        h1, obs_dim = state["trunk.0.weight"].shape
        h2 = state["trunk.2.weight"].shape[0]
        action_dim = state["advantage_head.weight"].shape[0]
    except KeyError as exc:  # not a dueling actor checkpoint
        raise ValueError(
            f"{pt_path.name}: unexpected state_dict layout (missing {exc})"
        ) from exc
    param_count = int(sum(t.numel() for t in state.values()))
    return {
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "net_arch": [int(h1), int(h2)],
        "param_count": param_count,
    }


def _build_card(pt_path: Path) -> dict[str, Any]:
    """Construct a fresh registry card for one checkpoint file."""
    stem = pt_path.stem
    arch = _infer_arch(pt_path)
    name = pt_path.name.lower()
    algo = "ppo" if "ppo" in name else "dqn"

    # Legacy checkpoints have no embedded HardwareSpec; tag the ones whose
    # dims match the default contract so they can be offered for finetuning.
    matches_default = (
        arch["obs_dim"] == _DEFAULT_OBS_DIM
        and arch["action_dim"] == _DEFAULT_ACTION_DIM
    )
    signature = _DEFAULT_SIGNATURE if matches_default else None

    return {
        "id": stem,
        "filename": pt_path.name,
        "algo": algo,
        "obs_dim": arch["obs_dim"],
        "action_dim": arch["action_dim"],
        "net_arch": arch["net_arch"],
        "param_count": arch["param_count"],
        "obs_signature_hash": signature,
        "metrics": None,  # filled by evaluate_model
        "created_at": _iso(pt_path.stat().st_mtime),
    }


def _card_path(model_id: str) -> Path:
    return config.REGISTRY_DIR / f"{model_id}.json"


def _read_card(model_id: str) -> dict[str, Any] | None:
    path = _card_path(model_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # corrupt cache -> treat as missing, will recompute


def _write_card(card: dict[str, Any]) -> None:
    config.ensure_dirs()
    _card_path(card["id"]).write_text(
        json.dumps(card, indent=2, sort_keys=True), encoding="utf-8"
    )


def _card_is_fresh(card: dict[str, Any], pt_path: Path) -> bool:
    """A cached card is fresh iff it's newer than (or equal to) its .pt."""
    cache_path = _card_path(card["id"])
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= pt_path.stat().st_mtime


def _checkpoint_for(model_id: str) -> Path | None:
    pt = config.CHECKPOINTS / f"{model_id}.pt"
    return pt if pt.exists() else None


def scan_checkpoints() -> list[dict[str, Any]]:
    """Build (or load from cache) a card for every ``checkpoints/*.pt``.

    Recomputes any card whose ``.pt`` is newer than the cached JSON.
    Checkpoints that aren't the dueling layout are skipped. Returns cards
    sorted by id.
    """
    config.ensure_dirs()
    cards: list[dict[str, Any]] = []
    for pt_path in sorted(config.CHECKPOINTS.glob("*.pt")):
        model_id = pt_path.stem
        cached = _read_card(model_id)
        if cached is not None and _card_is_fresh(cached, pt_path):
            cards.append(cached)
            continue
        try:
            card = _build_card(pt_path)
        except ValueError:
            continue  # not a recognised actor checkpoint
        # Preserve previously-computed metrics across a rebuild if the dims
        # are unchanged (the .pt was merely touched / re-saved).
        if cached is not None and cached.get("metrics") is not None and \
                cached.get("obs_dim") == card["obs_dim"] and \
                cached.get("action_dim") == card["action_dim"]:
            card["metrics"] = cached["metrics"]
        _write_card(card)
        cards.append(card)
    # 'protected' is a runtime property (not persisted), so it's always correct
    # even for cards loaded from an older cache.
    for c in cards:
        c["protected"] = c.get("id") in PROTECTED_MODEL_IDS
    return cards


def list_models() -> list[dict[str, Any]]:
    """All registry cards (one per recognised checkpoint)."""
    return scan_checkpoints()


def get_model(model_id: str) -> dict[str, Any] | None:
    """The card for ``model_id`` (filename stem), or ``None`` if unknown."""
    pt_path = _checkpoint_for(model_id)
    if pt_path is None:
        return None
    cached = _read_card(model_id)
    if cached is not None and _card_is_fresh(cached, pt_path):
        card = cached
    else:
        try:
            card = _build_card(pt_path)
        except ValueError:
            return None
        if cached is not None and cached.get("metrics") is not None and \
                cached.get("obs_dim") == card["obs_dim"] and \
                cached.get("action_dim") == card["action_dim"]:
            card["metrics"] = cached["metrics"]
        _write_card(card)
    card["protected"] = model_id in PROTECTED_MODEL_IDS
    return card


def delete_model(model_id: str) -> bool:
    """Delete a checkpoint and its cached registry card.

    Returns ``False`` if no such checkpoint exists. Raises ``ValueError`` for a
    protected (deployed/canonical) model — those are committed and must be
    removed deliberately via git, not the UI.
    """
    pt_path = _checkpoint_for(model_id)
    if pt_path is None:
        return False
    if model_id in PROTECTED_MODEL_IDS:
        raise ValueError(f"{model_id} is a protected/deployed model")
    pt_path.unlink(missing_ok=True)
    _card_path(model_id).unlink(missing_ok=True)
    return True


def finetune_candidates(
    signature: str | None, action_dim: int
) -> list[dict[str, Any]]:
    """Cards that are obs/action byte-compatible with ``signature``.

    A candidate matches when its ``obs_signature_hash`` equals ``signature``
    AND its ``action_dim`` matches. A ``None`` signature never matches (legacy
    checkpoints with no contract are not offered).
    """
    if signature is None:
        return []
    return [
        c
        for c in list_models()
        if c.get("obs_signature_hash") == signature
        and c.get("action_dim") == action_dim
    ]


# Opponent set + multiplier used by the on-demand evaluator. Kept small so a
# single evaluate call is bearable; it is never invoked from the test suite.
_EVAL_OPPONENTS = ("dodger", "spinner", "rammer")
_EVAL_MULT = 3.0
_EVAL_N = 10
_EVAL_SEED = 4242


def evaluate_model(model_id: str, mode: str = "quick") -> dict[str, Any] | None:
    """Run headless rollouts for ``model_id`` and cache the metrics.

    ``mode`` selects the opponent set:
      * ``"quick"`` — a fast 3-opponent probe (dodger/spinner/rammer).
      * ``"full"``  — the deployment-style gauntlet over the whole
        trained-against zoo PLUS the held-out opponents (incl. novamax), so
        the win-rate reflects real strength, not just easy evaders.
    Both run at mult 3.0, n=10 each. Writes the aggregated win-rate +
    self-out-rate + per-opponent metrics into the card. ``None`` for an
    unknown id.

    Slow (real physics rollouts) — call on demand only.
    """
    pt_path = _checkpoint_for(model_id)
    if pt_path is None:
        return None

    card = get_model(model_id)
    if card is None:
        return None

    # Import the model class + evaluator lazily so merely listing models never
    # pulls in the full training stack / PyBullet.
    from train_dqn_3d import DuelingQNet
    from scripts.eval_best import run_eval

    if mode == "full":
        from scripts.eval_best import SEEN_OPPONENTS, HELD_OUT
        opponents = tuple(SEEN_OPPONENTS) + tuple(HELD_OUT)
    else:
        opponents = _EVAL_OPPONENTS

    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    h1, obs_dim = state["trunk.0.weight"].shape
    h2 = state["trunk.2.weight"].shape[0]
    n_act = state["advantage_head.weight"].shape[0]
    model = DuelingQNet(int(obs_dim), int(n_act), hidden=(int(h1), int(h2)))
    model.load_state_dict(state)
    model.eval()

    per_opp: dict[str, Any] = {}
    total_wins = total_n = total_self = 0
    # PyBullet is single-client per process: serialize the rollouts so two
    # concurrent /evaluate calls queue instead of corrupting each other (500).
    with pybullet_lock:
        for opp in opponents:
            r = run_eval(model, opp, _EVAL_MULT, _EVAL_N, _EVAL_SEED)
            per_opp[opp] = {
                "wr": r["wr"],
                "wins": r["wins"],
                "losses": r["losses"],
                "timeouts": r["timeouts"],
                "self_out": r["self_out"],
                "mean_ep_len": r["mean_ep_len"],
            }
            total_wins += r["wins"]
            total_n += r["n"]
            total_self += r["self_out"]

    card["metrics"] = {
        "mode": mode,
        "win_rate": total_wins / total_n if total_n else 0.0,
        "self_outs": total_self,
        "self_out_rate": total_self / total_n if total_n else 0.0,
        "n_episodes": total_n,
        "mult": _EVAL_MULT,
        "opponents": list(opponents),
        "per_opponent": per_opp,
        "evaluated_at": _iso(datetime.now(tz=timezone.utc).timestamp()),
    }
    _write_card(card)
    return card
