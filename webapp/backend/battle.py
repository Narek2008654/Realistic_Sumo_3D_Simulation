"""ARENA head-to-head battle backend (additive, in-process).

A *battle* runs a trained model (side A) against either another model
(``b_model_id``) or a zoo opponent (``b_opponent_id``) for ``rounds`` rounds,
aggregates win/loss stats, and records ONE representative round's kinematic
trajectory (the first decisive round) to ``data/battles/<id>/trajectory.json``
in the same format the frontend ``TrajectoryPlayer`` already consumes.

Runs in-process and is guarded by ``pybullet_lock`` (PyBullet is single-client
per process) so it works even while a training subprocess runs — it does NOT
go through the training job manager.

Two execution paths, both reusing existing harnesses so there is no duplicated
physics / obs logic:

  * **model vs model** — reuses :func:`scripts.agent_vs_agent.play_match`
    (same chassis, two greedy policies via the obs save/restore swap). We add a
    thin pose-recording wrapper for the trajectory round.
  * **model vs opponent** — reuses an :func:`scripts.eval_best.run_eval`-style
    greedy rollout against ``force_opponent_id`` and the eval_and_record
    trajectory recorder.

Per-robot hardware: an optional ``a_spec`` HardwareSpec is threaded into the
AGENT side via ``build_env(hardware_spec=...)``. A CUSTOM (saved) opponent now
fights on ITS OWN chassis/motors — its saved HardwareSpec is threaded into the
env as ``enemy_hardware_spec`` so the enemy body is generated from it and its
drive caps come from spec.drivetrain. A standalone ``b_spec`` (B-side override
for model-vs-model) is still acknowledged in ``notes`` rather than applied.

Windows DLL-order convention: import ``torch`` before ``numpy``.
"""

from __future__ import annotations

import torch  # noqa: F401  (must precede numpy for Windows DLL ordering)

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from webapp.backend import config
from webapp.backend.pybullet_lock import pybullet_lock
from webapp.backend.registry import _checkpoint_for
from webapp.shared.hardware_spec import HardwareSpec

__all__ = ["run_battle", "BattleError"]


class BattleError(Exception):
    """Raised for caller-facing battle errors (mapped to HTTP status by the
    router). ``status`` is the HTTP code to surface."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _load_policy(model_id: str):
    """Load a model_id's checkpoint into a DuelingQNet (arch inferred from the
    weights, exactly as eval_best / registry do). Raises BattleError(404)."""
    pt_path = _checkpoint_for(model_id)
    if pt_path is None:
        raise BattleError(404, f"unknown model: {model_id}")
    from train_dqn_3d import DuelingQNet

    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    h1, obs_dim = state["trunk.0.weight"].shape
    h2 = state["trunk.2.weight"].shape[0]
    n_act = state["advantage_head.weight"].shape[0]
    net = DuelingQNet(int(obs_dim), int(n_act), hidden=(int(h1), int(h2)))
    net.load_state_dict(state)
    net.eval()
    return net


def _pose(p, body_id) -> dict[str, list[float]]:
    pos, orn = p.getBasePositionAndOrientation(body_id)
    return {"p": [float(v) for v in pos], "q": [float(v) for v in orn]}


# ---------------------------------------------------------------------------
# model vs zoo-opponent
# ---------------------------------------------------------------------------
def _battle_vs_opponent(
    net_a, opponent_id: str, rounds: int, mult: float, seed: int,
    hw_spec: HardwareSpec | None,
    extra_opponents: dict | None = None,
    enemy_hw_spec: HardwareSpec | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Run ``rounds`` greedy rollouts of model A vs a zoo or CUSTOM opponent.

    Returns ``(stats, round_trajectories)`` where ``round_trajectories`` is a
    list (one per round, in order) of ``{trajectory, winner, reason}``. Every
    round's full kinematics are captured so any round can be replayed.
    Mirrors ``eval_best.run_eval`` but per-round so we map outcomes to A/B wins.

    A built-in zoo id must exist in ``OPPONENT_REGISTRY``; a custom id must be
    supplied via ``extra_opponents`` (id -> controller factory), which is
    merged into the env's sampling pool and pinned via ``force_opponent_id``.

    ``enemy_hw_spec`` (custom opponents only) makes the enemy fight on its OWN
    chassis/motors: it is threaded into the env so the body is generated from
    that HardwareSpec and the enemy's drive caps come from spec.drivetrain.
    """
    from train_dqn_3d import build_env
    from webapp.shared.eval_and_record import _record_trajectory

    is_custom = bool(extra_opponents) and opponent_id in extra_opponents
    if not is_custom and opponent_id not in __import__("opponents").OPPONENT_REGISTRY:
        raise BattleError(422, f"unknown opponent: {opponent_id}")

    stats = Counter()
    round_trajs: list[dict[str, Any]] = []

    for r in range(rounds):
        env = build_env(
            gui=False, seed=seed + r,
            novamax_torque_mult=mult, force_opponent_id=opponent_id,
            narek_reward=False,
            **({"extra_opponents": extra_opponents} if extra_opponents else {}),
            **({"hardware_spec": hw_spec} if hw_spec is not None else {}),
            **({"enemy_hardware_spec": enemy_hw_spec}
               if enemy_hw_spec is not None else {}),
        )
        try:
            traj = _record_trajectory(env, net_a)
        finally:
            env.close()

        reason = traj["outcome"]["reason"]
        winner = traj["outcome"]["winner"]
        if reason == "win":
            stats["a_wins"] += 1
        elif reason == "timeout":
            stats["timeouts"] += 1
        elif reason == "self_out":
            stats["b_wins"] += 1   # A drove itself off -> B wins
            stats["a_self_out"] += 1
        elif reason in ("push_loss", "mutual_out"):
            if reason == "mutual_out":
                stats["draws"] += 1
            else:
                stats["b_wins"] += 1  # A pushed out -> B wins
        else:
            stats["draws"] += 1

        round_trajs.append(
            {"trajectory": traj, "winner": winner, "reason": reason}
        )

    return dict(stats), round_trajs


# ---------------------------------------------------------------------------
# model vs model (same chassis)
# ---------------------------------------------------------------------------
def _record_match(env_u, net_a, net_b, max_steps: int = 600):
    """Run ONE model-vs-model round and capture per-frame poses.

    Mirrors :func:`scripts.agent_vs_agent.play_match` step-for-step but also
    records the agent (A) + enemy (B) base poses each frame, so there is no
    duplicated *obs* logic (the obs swap helpers are imported, not re-written).
    Returns ``(winner, exit_kind, trajectory)`` where ``winner`` in
    {"A","B","draw","timeout"}.
    """
    import pybullet as p
    from sumo_env import (
        DISCRETE_ACTION_MAP, SUBSTEPS_PER_STEP, FALL_Z, STEP_DT_SECONDS,
    )
    from scripts.agent_vs_agent import fresh_state, robot_obs, drive

    contact_window = 10
    stA, stB = fresh_state(), fresh_state()
    obsA = robot_obs(env_u, env_u.robot_id, stA, 0.0, 0.0, first=True)
    obsB = robot_obs(env_u, env_u.enemy_id, stB, 0.0, 0.0, first=True)

    frames: list[dict] = [
        {"agent": _pose(p, env_u.robot_id), "enemy": _pose(p, env_u.enemy_id)}
    ]
    last_contact = -10_000
    winner, exit_kind = "timeout", ""
    for step in range(max_steps):
        rawA = DISCRETE_ACTION_MAP[net_a.act_greedy(obsA)]
        rawB = DISCRETE_ACTION_MAP[net_b.act_greedy(obsB)]
        drive(env_u._left_wheel_idx, env_u._right_wheel_idx, env_u.robot_id, *rawA)
        drive(env_u._enemy_left_idx, env_u._enemy_right_idx, env_u.enemy_id, *rawB)
        for _ in range(SUBSTEPS_PER_STEP):
            p.stepSimulation()
        frames.append(
            {"agent": _pose(p, env_u.robot_id), "enemy": _pose(p, env_u.enemy_id)}
        )
        if p.getContactPoints(bodyA=env_u.robot_id, bodyB=env_u.enemy_id):
            last_contact = step
        za = p.getBasePositionAndOrientation(env_u.robot_id)[0][2]
        zb = p.getBasePositionAndOrientation(env_u.enemy_id)[0][2]
        a_out, b_out = za < FALL_Z, zb < FALL_Z
        if a_out or b_out:
            if a_out and b_out:
                winner, exit_kind = "draw", "mutual"
            else:
                exit_kind = "push" if (step - last_contact) <= contact_window else "self_out"
                winner = "B" if a_out else "A"
            break
        obsA = robot_obs(env_u, env_u.robot_id, stA, *rawA)
        obsB = robot_obs(env_u, env_u.enemy_id, stB, *rawB)

    # Map to the frontend trajectory outcome vocabulary (agent == A).
    if winner == "A":
        out = {"winner": "agent", "reason": "win"}
    elif winner == "B":
        out = {"winner": "enemy",
               "reason": "self_out" if exit_kind == "self_out" else "push_loss"}
    elif winner == "draw":
        out = {"winner": None, "reason": "mutual_out"}
    else:
        out = {"winner": None, "reason": "timeout"}

    trajectory = {
        "dt": float(STEP_DT_SECONDS),
        # Actual ring size for this battle (side A's spec), so a resized dohyo
        # replays at the right size — not the module default.
        "dohyo_radius": float(env_u.hw_spec.dohyo.radius_m),
        "frames": frames,
        "outcome": out,
    }
    return winner, exit_kind, trajectory


def _battle_vs_model(
    net_a, net_b, rounds: int, mult: float, seed: int,
    hw_spec: HardwareSpec | None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Model-vs-model over ``rounds`` rounds on the SAME chassis.

    Reuses ``agent_vs_agent``'s env construction + faithfulness convention:
    both robots spawn on the agent URDF (``respawn_enemy_as_agent_robot``) so
    the match isolates the policies. Records every round.

    Returns ``(stats, round_trajectories)`` where ``round_trajectories`` is a
    list (one per round) of ``{trajectory, winner, reason}`` (winner/reason in
    the frontend trajectory vocabulary, taken from the trajectory's outcome).
    """
    import sumo_env
    from train_dqn_3d import build_env
    from scripts.agent_vs_agent import respawn_enemy_as_agent_robot

    # Deterministic + fair: disable per-step ToF noise for all resets, exactly
    # as the agent_vs_agent harness does.
    saved_noise = sumo_env.DR_TOF_NOISE_SIGMA_PCT
    sumo_env.DR_TOF_NOISE_SIGMA_PCT = 0.0
    env = build_env(
        gui=False, seed=seed, narek_reward=False,
        **({"hardware_spec": hw_spec} if hw_spec is not None else {}),
    )
    env_u = env.unwrapped

    stats = Counter()
    round_trajs: list[dict[str, Any]] = []
    try:
        for r in range(rounds):
            env.reset(seed=seed + r)
            respawn_enemy_as_agent_robot(env_u)
            winner, exit_kind, traj = _record_match(env_u, net_a, net_b)
            if winner == "A":
                stats["a_wins"] += 1
            elif winner == "B":
                stats["b_wins"] += 1
                if exit_kind == "self_out":
                    stats["a_self_out"] += 1
            elif winner == "draw":
                stats["draws"] += 1
            else:
                stats["timeouts"] += 1
            round_trajs.append({
                "trajectory": traj,
                "winner": traj["outcome"]["winner"],
                "reason": traj["outcome"]["reason"],
            })
    finally:
        env.close()
        sumo_env.DR_TOF_NOISE_SIGMA_PCT = saved_noise

    return dict(stats), round_trajs


# ---------------------------------------------------------------------------
# public entrypoint
# ---------------------------------------------------------------------------
def _full_stats(rounds: int, raw: dict[str, int]) -> dict[str, int]:
    """Normalize a Counter into the documented stats shape."""
    return {
        "rounds": rounds,
        "a_wins": raw.get("a_wins", 0),
        "b_wins": raw.get("b_wins", 0),
        "draws": raw.get("draws", 0),
        "timeouts": raw.get("timeouts", 0),
        "a_self_out": raw.get("a_self_out", 0),
        "b_self_out": raw.get("b_self_out", 0),
    }


def _representative_index(round_trajs: list[dict[str, Any]]) -> int:
    """Index of the first decisive round (reason != timeout), else 0."""
    for i, rt in enumerate(round_trajs):
        if rt["reason"] != "timeout":
            return i
    return 0


def _write_rounds(
    battle_dir: Path, round_trajs: list[dict[str, Any]],
    opponent_id: str | None, prefix: str = "",
) -> list[dict[str, Any]]:
    """Persist every round's trajectory to ``<battle_dir>/<prefix>r<k>.json`` and
    return a list of round summaries ``{index, opponent_id, winner, reason,
    trajectory_ref}``. ``prefix`` namespaces gauntlet rounds per opponent so
    refs never collide across opponents."""
    summaries: list[dict[str, Any]] = []
    for k, rt in enumerate(round_trajs):
        ref = f"{prefix}r{k}"
        (battle_dir / f"{ref}.json").write_text(
            json.dumps(rt["trajectory"]), encoding="utf-8"
        )
        summaries.append({
            "index": k,
            "opponent_id": opponent_id,
            "winner": rt["winner"],
            "reason": rt["reason"],
            "trajectory_ref": ref,
        })
    return summaries


def _resolve_custom_opponent(
    opponent_id: str, notes: list[str]
) -> tuple[dict, HardwareSpec] | None:
    """If ``opponent_id`` names a saved custom opponent, return
    ``(extra_opponents, enemy_hw_spec)`` where ``extra_opponents`` maps
    ``{id: DslOpponent factory}`` and ``enemy_hw_spec`` is the opponent's saved
    HardwareSpec; else None (a built-in zoo id, handled by
    ``_battle_vs_opponent`` directly).

    The custom opponent now fights on ITS OWN chassis/motors: the returned
    spec is threaded into the env as ``enemy_hardware_spec`` so the enemy body
    is generated from it and its drive caps come from spec.drivetrain. A note
    is appended recording that the custom hardware is in effect.
    """
    from webapp.backend import opponents_store

    record = opponents_store.get_opponent(opponent_id)
    if record is None:
        return None  # not a custom id -> let the zoo path validate it.

    from opponents.dsl_runtime import DslOpponent
    from webapp.shared.opponent_dsl import OpponentDSL

    dsl = OpponentDSL.from_dict(record["behavior_dsl"])
    enemy_spec = HardwareSpec.from_dict(record["hardware_spec"])
    notes.append(
        "Custom opponent fights on its own hardware: the enemy chassis, "
        "wheels, wedge, mass and motor caps come from its saved HardwareSpec "
        "(its behavior_dsl still drives the controller)."
    )
    return {opponent_id: lambda: DslOpponent(dsl)}, enemy_spec


def _gauntlet_opponent_ids(
    include_held_out: bool, include_custom: bool, notes: list[str],
) -> list[tuple[str, dict | None, HardwareSpec | None]]:
    """Resolve the gauntlet opponent roster as a list of
    ``(opponent_id, extra_opponents, enemy_hw_spec)``.

    The base roster is the SEEN zoo (built-in zoo minus held-out). With
    ``include_held_out`` the held-out opponents (feinter/orbiter) are appended.
    With ``include_custom`` every saved custom opponent is appended (each
    resolved to its DslOpponent factory + own hardware)."""
    import opponents as _opp

    seen = [oid for oid in _opp.OPPONENT_IDS
            if oid not in _opp.HELD_OUT_OPPONENT_IDS]
    roster: list[tuple[str, dict | None, HardwareSpec | None]] = [
        (oid, None, None) for oid in seen
    ]
    if include_held_out:
        roster += [(oid, None, None) for oid in _opp.HELD_OUT_OPPONENT_IDS]

    if include_custom:
        from webapp.backend import opponents_store
        custom_notes: list[str] = []
        for summary in opponents_store.list_opponents():
            cid = summary["id"]
            resolved = _resolve_custom_opponent(cid, custom_notes)
            if resolved is not None:
                extra, enemy_spec = resolved
                roster.append((cid, extra, enemy_spec))
        if custom_notes:
            notes.append(
                "Custom opponents fight on their own saved hardware."
            )
    return roster


def _gauntlet(
    net_a, roster: list[tuple[str, dict | None, HardwareSpec | None]],
    rounds: int, mult: float, seed: int, a_spec: HardwareSpec | None,
    battle_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run model A against every opponent in ``roster`` for ``rounds`` rounds.

    Writes each opponent's round trajectories under
    ``<battle_dir>/<opponent_id>__r<k>.json`` and returns
    ``(per_opponent, overall_raw)`` where ``per_opponent`` is a list of
    ``{opponent_id, stats, rounds:[...]}`` and ``overall_raw`` is the summed
    Counter across all opponents. Runs under the caller's PyBullet lock."""
    per_opponent: list[dict[str, Any]] = []
    overall = Counter()
    for opponent_id, extra, enemy_spec in roster:
        raw, round_trajs = _battle_vs_opponent(
            net_a, opponent_id, rounds, mult, seed, a_spec,
            extra_opponents=extra, enemy_hw_spec=enemy_spec,
        )
        # Namespace refs per-opponent so they never collide across opponents.
        summaries = _write_rounds(
            battle_dir, round_trajs, opponent_id,
            prefix=f"{opponent_id}__",
        )
        for k in raw:
            overall[k] += raw[k]
        per_opponent.append({
            "opponent_id": opponent_id,
            "stats": _full_stats(rounds, raw),
            "rounds": summaries,
        })
    return per_opponent, dict(overall)


def run_battle(req: dict[str, Any]) -> dict[str, Any]:
    """Run a head-to-head battle (single) or a whole-zoo gauntlet and persist
    every round's trajectory.

    ``req`` keys: ``a_model_id`` (required), ``mode`` ("single"|"gauntlet",
    default "single"), ``rounds`` (default 5), ``mult`` (default 3.0),
    ``seed`` (default 4242), optional ``a_spec`` / ``b_spec`` HardwareSpec
    dicts.

    * **single** — exactly one of ``b_model_id`` / ``b_opponent_id``. Returns
      ``{battle_id, mode:"single", stats, rounds, trajectory, notes?}`` where
      ``rounds`` is a per-round list ``[{index, opponent_id, winner, reason,
      trajectory_ref}]`` and ``trajectory`` is the first decisive round (for
      back-compat).
    * **gauntlet** — side B is ignored; A fights each built-in (seen) zoo
      opponent. ``include_held_out`` (default false) adds feinter/orbiter;
      ``include_custom`` (default false) adds saved custom opponents. Returns
      ``{battle_id, mode:"gauntlet", per_opponent, overall_stats, notes?}``.

    Raises BattleError for 422 (bad request) / 404 (unknown model).
    """
    a_model_id = req.get("a_model_id")
    if not a_model_id:
        raise BattleError(422, "a_model_id is required")

    mode = req.get("mode", "single")
    if mode not in ("single", "gauntlet"):
        raise BattleError(422, f"unknown mode: {mode}")

    rounds = int(req.get("rounds", 5))
    if rounds < 1:
        raise BattleError(422, "rounds must be >= 1")
    mult = float(req.get("mult", 3.0))
    seed = int(req.get("seed", 4242))

    a_spec_d = req.get("a_spec")
    b_spec_d = req.get("b_spec")
    a_spec = HardwareSpec.from_dict(a_spec_d) if a_spec_d else None

    notes: list[str] = []
    if b_spec_d is not None:
        notes.append(
            "b_spec was provided but per-side (B) hardware is not supported in "
            "v1: the B side runs on the default chassis. Only the A (agent) "
            "side honors a_spec."
        )

    battle_id = uuid.uuid4().hex[:12]
    config.ensure_dirs()
    battle_dir = config.BATTLES_DIR / battle_id
    battle_dir.mkdir(parents=True, exist_ok=True)

    if mode == "gauntlet":
        return _run_gauntlet(
            a_model_id, rounds, mult, seed, a_spec,
            include_held_out=bool(req.get("include_held_out", False)),
            include_custom=bool(req.get("include_custom", False)),
            battle_id=battle_id, battle_dir=battle_dir, notes=notes,
        )

    # ---- single ----------------------------------------------------------
    b_model_id = req.get("b_model_id")
    b_opponent_id = req.get("b_opponent_id")
    if bool(b_model_id) == bool(b_opponent_id):
        raise BattleError(
            422, "provide exactly one of b_model_id or b_opponent_id"
        )

    # Resolve a CUSTOM opponent id (vs a built-in zoo id) up front, outside the
    # PyBullet lock. A custom id is one saved via opponents_store; we build a
    # DslOpponent factory from its behavior_dsl and hand it to the env's
    # extra_opponents seam. Built-in zoo ids leave extra_opponents None.
    extra_opponents: dict | None = None
    enemy_hw_spec: HardwareSpec | None = None
    if b_opponent_id:
        resolved = _resolve_custom_opponent(b_opponent_id, notes)
        if resolved is not None:
            extra_opponents, enemy_hw_spec = resolved

    net_a = _load_policy(a_model_id)

    # Single-client PyBullet: hold the lock for the whole sim section so the
    # battle queues safely behind any in-process eval / URDF validation and
    # never collides with a concurrent PyBullet env in this process.
    with pybullet_lock:
        if b_model_id:
            net_b = _load_policy(b_model_id)
            raw, round_trajs = _battle_vs_model(
                net_a, net_b, rounds, mult, seed, a_spec
            )
        else:
            raw, round_trajs = _battle_vs_opponent(
                net_a, b_opponent_id, rounds, mult, seed, a_spec,
                extra_opponents=extra_opponents,
                enemy_hw_spec=enemy_hw_spec,
            )

    stats = _full_stats(rounds, raw)
    summaries = _write_rounds(battle_dir, round_trajs, b_opponent_id)

    # Back-compat: keep a top-level `trajectory` = the first decisive round.
    rep_idx = _representative_index(round_trajs)
    trajectory = round_trajs[rep_idx]["trajectory"]
    (battle_dir / "trajectory.json").write_text(
        json.dumps(trajectory), encoding="utf-8"
    )

    result: dict[str, Any] = {
        "battle_id": battle_id,
        "mode": "single",
        "stats": stats,
        "rounds": summaries,
        "trajectory": trajectory,
    }
    if notes:
        result["notes"] = " ".join(notes)
    return result


def _run_gauntlet(
    a_model_id: str, rounds: int, mult: float, seed: int,
    a_spec: HardwareSpec | None, include_held_out: bool, include_custom: bool,
    battle_id: str, battle_dir: Path, notes: list[str],
) -> dict[str, Any]:
    """Gauntlet entrypoint: A vs the whole (seen) zoo, one block per opponent."""
    roster = _gauntlet_opponent_ids(include_held_out, include_custom, notes)
    if not roster:
        raise BattleError(422, "gauntlet roster is empty")

    net_a = _load_policy(a_model_id)
    with pybullet_lock:
        per_opponent, overall_raw = _gauntlet(
            net_a, roster, rounds, mult, seed, a_spec, battle_dir
        )

    total_rounds = rounds * len(roster)
    result: dict[str, Any] = {
        "battle_id": battle_id,
        "mode": "gauntlet",
        "per_opponent": per_opponent,
        "overall_stats": _full_stats(total_rounds, overall_raw),
    }
    if notes:
        result["notes"] = " ".join(notes)
    return result


_SAFE_REF = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def load_trajectory(battle_id: str) -> dict[str, Any] | None:
    """Read a battle's representative trajectory (``trajectory.json``), or
    ``None`` if missing. Back-compat for the single-trajectory route."""
    return _read_battle_json(battle_id, "trajectory")


def load_trajectory_ref(battle_id: str, ref: str) -> dict[str, Any] | None:
    """Read a specific round trajectory ``<battle_id>/<ref>.json``, or ``None``
    if missing / the ref is not a safe filename (no path traversal)."""
    if not ref or not all(c in _SAFE_REF for c in ref):
        return None
    return _read_battle_json(battle_id, ref)


def _read_battle_json(battle_id: str, ref: str) -> dict[str, Any] | None:
    # Guard against path traversal: battle ids are hex tokens.
    if not battle_id or not all(c in "0123456789abcdef" for c in battle_id):
        return None
    path = config.BATTLES_DIR / battle_id / f"{ref}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
