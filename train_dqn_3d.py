"""Dueling Double DQN with n-step returns and Polyak target.

Modern upgrades over the baseline DQN at
github.com/Narek2008654/Simulator:
  * Dueling architecture (V + A heads, sharing a [48, 48] trunk)
  * Double DQN target (online net picks action, target net evaluates it)
  * n-step returns (n=3) for faster credit assignment
  * Polyak-averaged target network (tau=0.005) instead of hard copy
  * Multi-opponent offline pretrain on winner trajectories, then
    online fine-tune across a torque/opponent curriculum

Outputs:
  sac_actor_dqn.zip-equivalent weights via DQN-specific .pt + .npz
  neural_net_v6_3d.h via export_weights.py (argmax(advantage) for AVR
  inference; V head can be dropped since argmax over Q is invariant
  to the mean-zero shift dueling uses).

The agent uses MiniSumoEnv with action_space_kind='discrete' (9 actions
on the 3x3 motor grid) and narek_reward=True (idle/retreat/attack
shaping on top of the existing approach/engage/wedge/terminal stack).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch FIRST on Windows: importing numpy first can trigger a
# DLL-search-order race against torch's fbgemm.dll (WinError 127).
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from sumo_env import (
    DISCRETE_ACTION_MAP,
    MiniSumoEnv,
)
from obs_stack import RawDistanceStack, stacked_dim, DEFAULT_STACK_K, ENGINEERED_DIM
from combat_policy import CombatPolicy

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
# alg/improvment: the base env still emits a 12-D obs; the policy network
# consumes the K-frame raw-distance stack (RawDistanceStack) = 21-D for
# K=4. NET_OBS_DIM is the actual network input width.
OBS_DIM = 12
STACK_K = DEFAULT_STACK_K
NET_OBS_DIM = stacked_dim(STACK_K)        # 3*K + 9 = 21 for K=4
N_ACTIONS = 9
# 32x32: the deployable Nano-friendly size (~2k params, ~8 KB flash, fits
# the 24 Hz budget in float32) and the size of the best model so far
# (Stage-A). 48x48 did not beat it at mult 3.0.
NET_ARCH = (32, 32)

# Smoke mode (DQN_SMOKE=1): tiny collection + a few thousand online steps
# to validate the whole pipeline end-to-end before the multi-hour run.
SMOKE = bool(os.environ.get("DQN_SMOKE"))

# Run 15: continue training from Run-14's v2 model, adding the new
# action-persistence (anti-flicker) reward on top of the tracking
# reward already learned.
HERE_TOP = Path(__file__).parent
CHECKPOINTS = HERE_TOP / "checkpoints"; CHECKPOINTS.mkdir(exist_ok=True)
DATA = HERE_TOP / "data";               DATA.mkdir(exist_ok=True)
FIRMWARE = HERE_TOP / "firmware";       FIRMWARE.mkdir(exist_ok=True)

CONTINUE_TRAINING = False
RESUME_CHECKPOINT_PATH = CHECKPOINTS / "dqn_actor_v2_best.pt"
CONTINUE_BEST_PATH = CHECKPOINTS / "dqn_3d_actor_best.pt"
CONTINUE_FINAL_PATH = CHECKPOINTS / "dqn_3d_actor_final.pt"
# Reward flags now ported into 3D env (see sumo_env.py). Same recipe
# that took 2D from 0% trackers to rammer=44/novamax=26/wedger=19 at
# Phase-1 peak.
TRACKING_REWARD = True
ACTION_CONSISTENCY_REWARD = True
# alg/improvment: hardcoded observable safety override (rear-edge reflex +
# anti-blind-charge) active during training so the policy co-adapts and the
# deployed model carries the same deterministic reflexes.
SAFETY_OVERRIDE = True

GAMMA = 0.99
N_STEP = 3
TAU = 0.005                # Polyak coefficient for target net soft update
BATCH_SIZE = 256
REPLAY_CAPACITY = 200_000

# alg/improvment v3: demonstration retention (DQfD-style). The winning BC
# transitions are kept in a persistent buffer and mixed into every online
# gradient step — a 1-step Bellman loss plus a supervised large-margin loss
# that keeps the winning action's Q above the others. This anchors the
# policy to the teacher's winning behavior so online exploration against
# aggressive opponents refines it instead of catastrophically forgetting it.
DEMO_BATCH = 128
DEMO_BELLMAN_WEIGHT = 1.0
DEMO_MARGIN_WEIGHT = 1.0
DEMO_MARGIN = 0.8

# Run 12: per-opponent collection program. Each entry is
# (opp_id, torque_mult, max_env_steps, n_pairs_target). Mults are
# tuned per opponent so the CombatPolicy actually wins enough episodes
# to produce demonstration pairs:
#   * dodger/spinner: passive evaders, easy at mult=0.7
#   * rammer/wedger/novamax/charger: aggressive trackers; orbital
#     deadlock at mult=0.7 means ~0% WR even with combat_policy.
#     Drop them to mult=0.3 (NOVAMAX_TORQUE_MULT_MIN floor) where the
#     opponent has near-zero pushing torque and the agent's wedge
#     dominates any contact.
# Online curriculum (later) still ramps mult back to 3.0 so the DQN
# learns to beat fully-powered opponents through online exploration.
OFFLINE_COLLECTION_PROGRAM: tuple[tuple[str, float, int, int], ...] = (
    # (opp_id,   mult,   max_steps,  n_pairs_target)
    # 3D physics post-fix — CombatPolicy now hits 85% vs novamax at
    # mult=1.0, so we can collect rich winning trajectories at the
    # mults the model will actually face during online curriculum.
    # All opponents pinned to mult=1.0 (matched torque) — keeps the
    # demonstration distribution close to the early Phase 1 the BC
    # warmstart needs to clear.
    ("dodger",   1.0,    20_000,    10_000),
    ("spinner",  1.0,    30_000,     6_000),
    ("rammer",   1.0,    30_000,     6_000),
    ("wedger",   1.0,    30_000,     6_000),
    ("novamax",  1.0,    40_000,     6_000),
    ("charger",  1.0,    30_000,     6_000),
)
OFFLINE_EPOCHS = 20
OFFLINE_LR = 1e-3

if CONTINUE_TRAINING:
    # Full 700k 3D fine-tune from v2_best — matches Run-14 schedule.
    # Smoke run already confirmed deadlock fix; this run lets the policy
    # actually adapt to PyBullet's wheel/contact physics.
    ONLINE_TIMESTEPS_PER_PHASE = (
        (1.0, 200_000, None),
        (2.0, 200_000, None),
        (3.0, 300_000, None),
    )
else:
    ONLINE_TIMESTEPS_PER_PHASE = (
        # (torque_mult, steps, force_opponent_id_or_None)
        # alg/improvment v3: dodger phase-1 restored (it builds clean wedge/
        # evasion mechanics on an easy opponent that generalize — removing it
        # hurt). Extended mult-3.0 phase: that is the eval/deploy target
        # (65% overall, 55% novamax @ mult 3).
        (0.7, 100_000, "dodger"),
        (1.0, 200_000, None),
        (1.5, 200_000, None),
        (2.0, 200_000, None),
        (3.0, 400_000, None),
    )
ONLINE_LR = 1e-4   # v3: was 3e-4 — conservative online updates so winner-only
                   # BC behavior is refined, not catastrophically forgotten.
# Run 13 (2D): slower / floored eps schedule + per-phase bump.
# EPS_DECAY_STEPS lengthened from 200k to 600k so exploration stays
# meaningful through the curriculum; EPS_END raised from 0.02 to 0.05
# so we never fully commit to greedy play. At each phase boundary
# we re-floor eps to PHASE_BUMP_EPS so every curriculum jump gets a
# fresh exploration window (otherwise zoo-phase trackers never get
# explored once eps has decayed too far).
if CONTINUE_TRAINING:
    # Model is already competent; need exploration mostly for the
    # new tracking reward, not for from-scratch learning.
    EPS_START = 0.15
    EPS_END = 0.03
    EPS_DECAY_STEPS = 400_000
    PHASE_BUMP_EPS = 0.12
else:
    # v3: lower exploration (was 0.30/0.05). Random play against aggressive
    # opponents floods replay with losses and erodes the BC winning behavior;
    # demo retention does the rest of the heavy lifting.
    EPS_START = 0.20
    EPS_END = 0.03
    EPS_DECAY_STEPS = 600_000
    PHASE_BUMP_EPS = 0.12
TRAIN_EVERY = 4            # gradient step every N env steps
TARGET_UPDATE_EVERY = 1    # Polyak soft update every step

HERE = HERE_TOP
OUTPUT_DIR = HERE / "checkpoints" / "dqn_intermediate"
# alg/improvment: new checkpoint names so the committed (deployed)
# baseline dqn_3d_bc_actor_best.pt is preserved for head-to-head eval.
BEST_PATH = CHECKPOINTS / "dqn3d_stable_best.pt"
FINAL_PATH = CHECKPOINTS / "dqn3d_stable_final.pt"
# (CONTINUE_TRAINING and related constants are declared above.)
# Collected winning dataset cached to disk so subsequent runs can
# skip Phase 0a entirely. Two sources are supported and concatenated
# when both exist:
#   * bc_dataset_v5.npz      -- scripted (CombatPolicy)
#   * bc_dataset_human.npz   -- human gameplay (human_play.py)
# Delete the relevant file to force fresh collection (e.g. if the
# env / reward / policy change). Human data is always preserved
# across scripted re-collections because it lives in a separate file.
# alg/improvment: 21-D stacked dataset (the v3/human sets are 12-D and
# incompatible — the loader skips any source whose width != NET_OBS_DIM).
BC_DATASET_PATH = DATA / "bc_dataset_3d_v6.npz"
BC_HUMAN_DATASET_PATH = DATA / "bc_dataset_human.npz"
LOG_EVERY = 5_000

# Smoke shrink: validate collect -> pretrain -> online -> export end-to-end
# in ~2 min. Must run BEFORE the function defs so OFFLINE_EPOCHS is captured
# in offline_pretrain's default arg.
if SMOKE:
    OFFLINE_COLLECTION_PROGRAM = (
        ("dodger", 1.0, 4_000, 400),
        ("novamax", 1.0, 4_000, 400),
    )
    ONLINE_TIMESTEPS_PER_PHASE = (
        (1.0, 3_000, "dodger"),
        (1.5, 3_000, None),
    )
    OFFLINE_EPOCHS = 2
    LOG_EVERY = 1_000

# alg/improvment: self-out-fix finetune (set DQN_FINETUNE=1). ~82% of
# losses at mult 3.0 are self-outs, so resume the best base model and
# train at mult 3.0 with edge-avoidance shaping ON over the BALANCED zoo
# (an earlier novamax-weighted mix caused catastrophic interference and
# regressed the strong opponents). Demo retention keeps the winning
# behavior; low eps refines rather than re-explores. Set DQN_FINETUNE_FROM
# to whichever base model (32x32 Stage-A or 48x48) evaluated best.
FINETUNE = bool(os.environ.get("DQN_FINETUNE"))
FINETUNE_RESUME = CHECKPOINTS / os.environ.get(
    "DQN_FINETUNE_FROM", "dqn3d_stack_stageA_best.pt",
)
FINETUNE_PHASES = ((3.0, 400_000, None),)
FINETUNE_EPS_START = 0.12
if SMOKE and FINETUNE:
    FINETUNE_PHASES = ((3.0, 3_000, None),)


# ---------------------------------------------------------------------------
# Env factory + checkpointing helpers
# ---------------------------------------------------------------------------
def build_env(**kwargs) -> RawDistanceStack:
    """Construct the discrete-action sumo env wrapped in the K-frame
    raw-distance stack. All policy-facing code sees the stacked obs."""
    kwargs.setdefault("action_space_kind", "discrete")
    return RawDistanceStack(MiniSumoEnv(**kwargs), k=STACK_K)


def save_checkpoint_atomic(state_dict, path: Path) -> None:
    """Write a checkpoint via a temp file + os.replace so a concurrent
    eval never reads a half-written file (Windows torch.save isn't atomic)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, str(tmp))
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Action quantization (continuous combat_policy output -> discrete index)
# ---------------------------------------------------------------------------
def quantize_to_idx(left: float, right: float) -> int:
    """Map a continuous (l, r) pair to the nearest discrete action index.

    Each motor command is snapped to {-1, 0, +1} by sign + magnitude:
      |x| < 0.5  -> 0
      x >= 0.5   -> +1
      x <= -0.5  -> -1
    Then the (l_snap, r_snap) pair is looked up in DISCRETE_ACTION_MAP.
    """
    def snap(x: float) -> float:
        if x >= 0.5:
            return 1.0
        if x <= -0.5:
            return -1.0
        return 0.0

    pair = (snap(left), snap(right))
    return DISCRETE_ACTION_MAP.index(pair)


# ---------------------------------------------------------------------------
# Dueling Q-network
# ---------------------------------------------------------------------------
class DuelingQNet(nn.Module):
    """Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)).

    Shared trunk feeds two heads:
      value_head:     trunk -> Linear(1)
      advantage_head: trunk -> Linear(N_ACTIONS)

    At deploy time, argmax over Q == argmax over A (V and the mean
    shift uniformly across actions), so the firmware only needs the
    advantage head — saves flash on AVR.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden=NET_ARCH):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, 1)
        self.advantage_head = nn.Linear(prev, n_actions)
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + a - a.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        """Argmax over Q. Uses advantage-only (V cancels under argmax)."""
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        h = self.trunk(x)
        a = self.advantage_head(h)
        return int(torch.argmax(a, dim=1).item())


# ---------------------------------------------------------------------------
# N-step replay buffer
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class NStepReplayBuffer:
    """Stores 1-step transitions and emits n-step samples on draw.

    Each draw returns (s_t, a_t, R_t^{(n)}, s_{t+n}, done_{t+n}) where
      R_t^{(n)} = sum_{k=0..n-1} gamma^k r_{t+k}
    and done is the OR of the n termination flags. If termination
    occurs at step t+k < n, the sum is truncated (no rewards after
    the terminal state) and done=True is returned with s_{t+k+1} as
    the (irrelevant under done=True) next-state placeholder.
    """

    def __init__(self, capacity: int, gamma: float, n_step: int):
        self.capacity = capacity
        self.gamma = gamma
        self.n_step = n_step
        # Ring buffer of Transitions.
        self.buf: list[Optional[Transition]] = [None] * capacity
        self.idx = 0
        self.size = 0
        # n_step staging deque: holds the most recent n transitions
        # from the CURRENT episode; cleared on episode end.
        self.staging: deque[Transition] = deque()

    def __len__(self) -> int:
        return self.size

    def _collapse_and_push(self) -> None:
        """Compute the n-step aggregate over self.staging and push the
        head transition (with aggregated reward/next/done) into the
        ring buffer. Pops the head from staging.
        """
        head = self.staging[0]
        reward = 0.0
        done = False
        next_obs = head.next_obs
        for k, tr in enumerate(self.staging):
            reward += (self.gamma ** k) * tr.reward
            next_obs = tr.next_obs
            done = tr.done
            if tr.done:
                break
        agg = Transition(
            obs=head.obs, action=head.action, reward=reward,
            next_obs=next_obs, done=done,
        )
        self.buf[self.idx] = agg
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.staging.popleft()

    def add(self, tr: Transition) -> None:
        self.staging.append(tr)
        # When staging is full (or the latest tr ends the episode),
        # collapse and push. On episode end, flush everything.
        if tr.done:
            while self.staging:
                self._collapse_and_push()
        elif len(self.staging) >= self.n_step:
            self._collapse_and_push()

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = [self.buf[i] for i in idxs]
        obs = np.stack([t.obs for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in batch]).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return obs, actions, rewards, next_obs, dones


# ---------------------------------------------------------------------------
# Loss / target computation
# ---------------------------------------------------------------------------
def double_dqn_loss(
    online: DuelingQNet,
    target: DuelingQNet,
    obs: torch.Tensor, actions: torch.Tensor,
    rewards: torch.Tensor, next_obs: torch.Tensor, dones: torch.Tensor,
    gamma_n: float,
) -> torch.Tensor:
    """Double DQN target with n-step gamma compounding.

    target = r_t + gamma^n * (1 - done) * Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a))
    """
    q_pred = online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = online(next_obs).argmax(dim=1, keepdim=True)
        next_q = target(next_obs).gather(1, next_actions).squeeze(1)
        td_target = rewards + gamma_n * (1.0 - dones) * next_q
    return F.smooth_l1_loss(q_pred, td_target)


def demo_margin_loss(
    online: DuelingQNet, obs: torch.Tensor, actions: torch.Tensor,
    margin: float = DEMO_MARGIN,
) -> torch.Tensor:
    """DQfD large-margin supervised loss: push Q(s, a_expert) to exceed
    every other action's Q by at least ``margin``. Zero once the expert
    action is already the (margin-separated) argmax, so it only corrects
    states where the policy has drifted off the demonstrated action.
    """
    q = online(obs)                                   # (B, A)
    rows = torch.arange(q.shape[0])
    margins = torch.full_like(q, margin)
    margins[rows, actions] = 0.0
    violation = (q + margins).max(dim=1).values - q[rows, actions]
    return violation.mean()


def polyak_update(online: DuelingQNet, target: DuelingQNet, tau: float) -> None:
    """target = tau*online + (1-tau)*target, in-place."""
    with torch.no_grad():
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


# ---------------------------------------------------------------------------
# Offline dataset collection (winner-only trajectories)
# ---------------------------------------------------------------------------
def collect_winning_dataset(
    program: tuple[tuple[str, float, int, int], ...],
    seed: int = SEED + 1,
    verbose: bool = True,
    explore_eps: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``CombatPolicy`` against each entry in ``program`` and keep
    transitions from WINNING episodes only.

    ``program`` items are ``(opp_id, torque_mult, max_env_steps,
    n_pairs_target)`` so each opponent can have its own difficulty and
    collection budget. The agent uses ``MiniSumoEnv`` in
    ``action_space_kind="discrete"`` with ``narek_reward=True`` so the
    per-step rewards stored alongside transitions include the same
    semantic shaping the online DQN will see.

    Returns concatenated arrays ``(obs, action_idx, reward, next_obs,
    done)`` across all opponents.
    """
    all_obs, all_act, all_rew, all_next, all_done = [], [], [], [], []
    for prog_idx, (opp, mult, max_steps, n_pairs) in enumerate(program):
        env = build_env(
            gui=False, seed=seed + prog_idx,
            novamax_torque_mult=mult,
            force_opponent_id=opp,
            narek_reward=True,
        )
        obs, _ = env.reset(seed=seed + prog_idx)
        policy = CombatPolicy()
        policy.reset()

        # Episode staging: keep pairs until we know the outcome, then
        # commit (win) or discard (loss / timeout).
        ep_obs, ep_act, ep_rew, ep_next, ep_done = [], [], [], [], []
        n_added = 0
        n_wins = 0
        n_episodes = 0
        n_attempts = 0
        while n_added < n_pairs and n_attempts < max_steps:
            n_attempts += 1
            # The teacher reads the CURRENT 12-D frame (tail of the stack);
            # epsilon-exploration occasionally substitutes a random action so
            # the winning set covers all 9 actions, not just the teacher's 3.
            l_cont, r_cont = policy(obs[-OBS_DIM:])
            if random.random() < explore_eps:
                a_idx = random.randrange(N_ACTIONS)
            else:
                a_idx = quantize_to_idx(l_cont, r_cont)
            next_obs, reward, terminated, truncated, info = env.step(a_idx)
            done = bool(terminated or truncated)

            ep_obs.append(obs.copy())
            ep_act.append(a_idx)
            ep_rew.append(float(reward))
            ep_next.append(next_obs.copy())
            ep_done.append(done)

            if done:
                n_episodes += 1
                if info.get("termination_reason") == "win":
                    n_wins += 1
                    take = min(len(ep_obs), n_pairs - n_added)
                    all_obs.extend(ep_obs[:take])
                    all_act.extend(ep_act[:take])
                    all_rew.extend(ep_rew[:take])
                    all_next.extend(ep_next[:take])
                    all_done.extend(ep_done[:take])
                    n_added += take
                ep_obs, ep_act, ep_rew, ep_next, ep_done = [], [], [], [], []
                obs, _ = env.reset()
                policy.reset()
            else:
                obs = next_obs
        env.close()
        if verbose:
            wr = n_wins / max(1, n_episodes)
            print(
                f"  [{opp:8s} mult={mult}] collected {n_added:5d} "
                f"winning pairs from {n_wins:3d}/{n_episodes:3d} eps "
                f"({wr:.0%} WR, {n_attempts:5d} env-steps used)",
                flush=True,
            )

    obs_arr = np.stack(all_obs).astype(np.float32) if all_obs else np.zeros(
        (0, NET_OBS_DIM), dtype=np.float32,
    )
    act_arr = np.array(all_act, dtype=np.int64)
    rew_arr = np.array(all_rew, dtype=np.float32)
    next_arr = np.stack(all_next).astype(np.float32) if all_next else np.zeros(
        (0, NET_OBS_DIM), dtype=np.float32,
    )
    done_arr = np.array(all_done, dtype=np.float32)
    if verbose:
        print(
            f"Winners dataset: {obs_arr.shape[0]} transitions total "
            f"across {len(program)} (opp, mult) configs",
            flush=True,
        )
    return obs_arr, act_arr, rew_arr, next_arr, done_arr


# ---------------------------------------------------------------------------
# Offline pretrain (Double DQN on the winner dataset)
# ---------------------------------------------------------------------------
def offline_pretrain(
    online: DuelingQNet,
    target: DuelingQNet,
    obs_arr: np.ndarray, act_arr: np.ndarray, rew_arr: np.ndarray,
    next_arr: np.ndarray, done_arr: np.ndarray,
    n_epochs: int = OFFLINE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
    lr: float = OFFLINE_LR,
    verbose: bool = True,
) -> None:
    if obs_arr.shape[0] == 0:
        print("offline_pretrain skipped: empty dataset", flush=True)
        return
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    n = obs_arr.shape[0]
    obs_t = torch.as_tensor(obs_arr)
    act_t = torch.as_tensor(act_arr)
    rew_t = torch.as_tensor(rew_arr)
    next_t = torch.as_tensor(next_arr)
    done_t = torch.as_tensor(done_arr)
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            loss = double_dqn_loss(
                online, target,
                obs_t[sel], act_t[sel], rew_t[sel],
                next_t[sel], done_t[sel],
                gamma_n=gamma,  # offline dataset is 1-step
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            opt.step()
            polyak_update(online, target, TAU)
            losses.append(float(loss.item()))
        if verbose:
            print(
                f"  offline epoch {epoch+1}/{n_epochs}  "
                f"loss={np.mean(losses):.4f} (min {np.min(losses):.4f}, "
                f"max {np.max(losses):.4f})",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Online fine-tune (epsilon-greedy + n-step replay + Double DQN)
# ---------------------------------------------------------------------------
def online_finetune(
    online: DuelingQNet,
    target: DuelingQNet,
    initial_mult: float,
    initial_opp: Optional[str],
    demos: Optional[tuple] = None,
    phases: Optional[tuple] = None,
    opponent_weights: Optional[dict] = None,
    eps_start: Optional[float] = None,
    edge_avoid: bool = False,
) -> dict:
    # alg/improvment v2: Stage A learns to fight under the STANDARD noise
    # regime (the same one the 40% baseline trained under) plus flank
    # shaping + frame-stack memory. Opponent-DR and hard-sensor-DR are OFF
    # here — piling them on from step 1 crippled perception of fast
    # opponents and the policy collapsed. They return in a Stage-B
    # robustness finetune once the clean win-rate target is met.
    # ``phases`` / ``opponent_weights`` / ``eps_start`` let a targeted
    # finetune override the curriculum, zoo mix, and exploration.
    if phases is None:
        phases = ONLINE_TIMESTEPS_PER_PHASE
    env = build_env(
        gui=False, seed=SEED,
        novamax_torque_mult=initial_mult,
        force_opponent_id=initial_opp,
        narek_reward=True,
        action_consistency_reward=ACTION_CONSISTENCY_REWARD,
        flank_reward=True,
        edge_avoid_reward=edge_avoid,
        safety_override=SAFETY_OVERRIDE,
        opponent_dr=False,
        sensor_hard_dr=False,
        opponent_weights=opponent_weights,
    )
    buffer = NStepReplayBuffer(REPLAY_CAPACITY, GAMMA, N_STEP)
    opt = torch.optim.Adam(online.parameters(), lr=ONLINE_LR)
    gamma_n = GAMMA ** N_STEP

    # Persistent demonstration tensors (winner-only BC transitions). Mixed
    # into every gradient step so the policy never forgets winning behavior.
    demo_n = 0
    if demos is not None and demos[0].shape[0] > 0:
        d_obs, d_act, d_rew, d_next, d_done = demos
        demo_obs_t = torch.as_tensor(d_obs)
        demo_act_t = torch.as_tensor(d_act)
        demo_rew_t = torch.as_tensor(d_rew)
        demo_next_t = torch.as_tensor(d_next)
        demo_done_t = torch.as_tensor(d_done)
        demo_n = d_obs.shape[0]
        print(f"  demo retention: {demo_n} winning transitions held", flush=True)

    obs, _ = env.reset(seed=SEED)
    step = 0
    episodes = 0
    recent_outcomes: deque[int] = deque(maxlen=200)
    per_opp_outcomes: dict[str, deque[int]] = {}
    best_score = -1.0
    eps = EPS_START if eps_start is None else eps_start

    # Per-step linear eps decay rate (constant). The decay is applied
    # incrementally so phase-bump resets stick instead of getting
    # overwritten by a step-indexed schedule.
    eps_step_decay = (eps - EPS_END) / EPS_DECAY_STEPS

    for phase_idx, (mult, n_steps, force_opp) in enumerate(
        phases, start=1,
    ):
        # Set on .unwrapped: attribute writes on a gym.Wrapper bind to the
        # wrapper, not the underlying env, so the curriculum knobs would
        # otherwise silently no-op.
        env.unwrapped.novamax_torque_mult = float(mult)
        env.unwrapped.force_opponent_id = force_opp
        # Reset the rolling WR window at each phase boundary: the
        # best-checkpoint score = wr * mult must measure WR against the
        # CURRENT phase opponent/mult, not stale outcomes carried over
        # from the prior (easier) phase. Clearing on phase 1 is a no-op.
        recent_outcomes.clear()
        # Phase bump: re-inject exploration on every curriculum
        # boundary so new opponents / mults get probed before the
        # decayed eps locks the policy into greedy play.
        if phase_idx > 1:
            eps = max(eps, PHASE_BUMP_EPS)
        print(
            f"\n=== DQN online phase {phase_idx}/"
            f"{len(phases)}: mult={mult}, "
            f"opp={force_opp or 'zoo'}, {n_steps} steps  "
            f"(eps={eps:.3f}) ===",
            flush=True,
        )
        phase_start = step
        last_log_step = step
        t_phase_start = time.time()
        while step - phase_start < n_steps:
            # epsilon-greedy action selection
            if random.random() < eps:
                a_idx = random.randrange(N_ACTIONS)
            else:
                a_idx = online.act_greedy(obs)
            next_obs, reward, terminated, truncated, info = env.step(a_idx)
            done = bool(terminated or truncated)
            buffer.add(Transition(
                obs=obs.copy(), action=a_idx, reward=float(reward),
                next_obs=next_obs.copy(), done=done,
            ))

            if done:
                episodes += 1
                reason = info.get("termination_reason", "")
                outcome = 1 if reason == "win" else 0
                recent_outcomes.append(outcome)
                opp_id = info.get("opponent_id", "unknown")
                bucket = per_opp_outcomes.setdefault(opp_id, deque(maxlen=200))
                bucket.append(outcome)
                obs, _ = env.reset()
            else:
                obs = next_obs

            step += 1
            # Per-step linear decay (stateful, so phase bumps stick).
            eps = max(EPS_END, eps - eps_step_decay)

            if step % TRAIN_EVERY == 0 and len(buffer) >= BATCH_SIZE:
                o, a, r, no_, d = buffer.sample(BATCH_SIZE)
                loss = double_dqn_loss(
                    online, target,
                    torch.as_tensor(o), torch.as_tensor(a),
                    torch.as_tensor(r), torch.as_tensor(no_),
                    torch.as_tensor(d), gamma_n=gamma_n,
                )
                if demo_n:
                    sel = np.random.randint(0, demo_n, size=DEMO_BATCH)
                    # Demos are 1-step transitions -> gamma^1, not gamma^n.
                    loss = loss + DEMO_BELLMAN_WEIGHT * double_dqn_loss(
                        online, target,
                        demo_obs_t[sel], demo_act_t[sel], demo_rew_t[sel],
                        demo_next_t[sel], demo_done_t[sel], gamma_n=GAMMA,
                    )
                    loss = loss + DEMO_MARGIN_WEIGHT * demo_margin_loss(
                        online, demo_obs_t[sel], demo_act_t[sel],
                    )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                opt.step()
                if step % TARGET_UPDATE_EVERY == 0:
                    polyak_update(online, target, TAU)

            if step - last_log_step >= LOG_EVERY:
                last_log_step = step
                wr = (sum(recent_outcomes) / len(recent_outcomes)
                      if recent_outcomes else 0.0)
                score = wr * mult
                marker = ""
                if score > best_score and len(recent_outcomes) >= 100:
                    best_score = score
                    save_path = (
                        CONTINUE_BEST_PATH if CONTINUE_TRAINING else BEST_PATH
                    )
                    save_checkpoint_atomic(online.state_dict(), save_path)
                    marker = "  *BEST*"
                opp_strs = []
                for o_id in sorted(per_opp_outcomes):
                    outs = per_opp_outcomes[o_id]
                    o_wr = sum(outs) / max(1, len(outs))
                    opp_strs.append(f"{o_id}={o_wr:.0%}({len(outs)})")
                fps = (step - phase_start) / max(1e-6, time.time() - t_phase_start)
                print(
                    f"step {step:7d}  ep={episodes:5d}  "
                    f"wr({len(recent_outcomes)})={wr:.2%}  "
                    f"eps={eps:.3f}  fps={fps:.1f}{marker}",
                    flush=True,
                )
                if opp_strs:
                    print("  per_opp: " + "  ".join(opp_strs), flush=True)

    env.close()
    final_path = CONTINUE_FINAL_PATH if CONTINUE_TRAINING else FINAL_PATH
    save_checkpoint_atomic(online.state_dict(), final_path)
    return {
        "total_steps": step,
        "total_episodes": episodes,
        "best_score": best_score,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(
        f"DQN run — net_arch={NET_ARCH}, n_actions={N_ACTIONS}, "
        f"n_step={N_STEP}, tau={TAU}",
        flush=True,
    )
    online = DuelingQNet(NET_OBS_DIM, N_ACTIONS, hidden=NET_ARCH)
    target = DuelingQNet(NET_OBS_DIM, N_ACTIONS, hidden=NET_ARCH)
    target.load_state_dict(online.state_dict())

    # alg/improvment: targeted novamax finetune. Resume the Stage-A best,
    # skip Phase 0, and run a mult-3.0 novamax-heavy online pass with demo
    # retention, then re-export. Best checkpoint still goes to BEST_PATH
    # (back it up before launching if you want to keep the Stage-A model).
    if FINETUNE:
        if not FINETUNE_RESUME.exists():
            raise SystemExit(
                f"DQN_FINETUNE=1 but resume checkpoint not found: "
                f"{FINETUNE_RESUME}"
            )
        state = torch.load(
            str(FINETUNE_RESUME), map_location="cpu", weights_only=True,
        )
        # Rebuild the net at the resume checkpoint's architecture so the
        # finetune works on either base (32x32 Stage-A or 48x48) regardless
        # of the module-level NET_ARCH.
        fh1, fobs = state["trunk.0.weight"].shape
        fh2 = state["trunk.2.weight"].shape[0]
        fnact = state["advantage_head.weight"].shape[0]
        online = DuelingQNet(fobs, fnact, hidden=(fh1, fh2))
        target = DuelingQNet(fobs, fnact, hidden=(fh1, fh2))
        online.load_state_dict(state)
        target.load_state_dict(state)
        cached = np.load(BC_DATASET_PATH)
        demos = (
            cached["obs"], cached["act"], cached["rew"],
            cached["next"], cached["done"],
        )
        print(
            f"\nFINETUNE (self-out fix): resume {FINETUNE_RESUME.name}, "
            f"mult-3.0 balanced zoo + edge-avoidance, "
            f"phases={FINETUNE_PHASES}, eps0={FINETUNE_EPS_START}",
            flush=True,
        )
        init_mult, _, init_opp = FINETUNE_PHASES[0]
        stats = online_finetune(
            online, target, init_mult, init_opp,
            demos=demos, phases=FINETUNE_PHASES,
            eps_start=FINETUNE_EPS_START, edge_avoid=True,
        )
        print(
            f"\nFinetune complete: {stats['total_episodes']} episodes, "
            f"best score {stats['best_score']:.3f}",
            flush=True,
        )
        if BEST_PATH.exists():
            online.load_state_dict(
                torch.load(str(BEST_PATH), map_location="cpu", weights_only=True)
            )
        emit_firmware_headers(online)
        return

    # Run 14: resume-from-checkpoint mode. Skip Phase 0a/0b entirely
    # and pick up online training from a previous best snapshot.
    if CONTINUE_TRAINING:
        if not RESUME_CHECKPOINT_PATH.exists():
            raise SystemExit(
                f"CONTINUE_TRAINING=True but checkpoint not found: "
                f"{RESUME_CHECKPOINT_PATH}"
            )
        print(
            f"\nResume mode: loading {RESUME_CHECKPOINT_PATH.name}, "
            f"skipping Phase 0a/0b, going straight to online fine-tune "
            f"with TRACKING_REWARD={TRACKING_REWARD} "
            f"ACTION_CONSISTENCY_REWARD={ACTION_CONSISTENCY_REWARD}",
            flush=True,
        )
        state = torch.load(
            str(RESUME_CHECKPOINT_PATH), map_location="cpu", weights_only=True,
        )
        online.load_state_dict(state)
        target.load_state_dict(state)
        initial_mult, _, initial_opp = ONLINE_TIMESTEPS_PER_PHASE[0]
        print(
            f"\nPhase 1+: online fine-tune (resume), curriculum starts at "
            f"mult={initial_mult}, opp={initial_opp or 'zoo'}",
            flush=True,
        )
        stats = online_finetune(online, target, initial_mult, initial_opp)
        print(
            f"\nDQN resume complete: {stats['total_episodes']} episodes, "
            f"best score {stats['best_score']:.3f}",
            flush=True,
        )
        best_path_to_export = CONTINUE_BEST_PATH
        if best_path_to_export.exists():
            online.load_state_dict(
                torch.load(str(best_path_to_export), map_location="cpu", weights_only=True)
            )
            print(f"Exporting BEST snapshot from {best_path_to_export.name}.",
                  flush=True)
        emit_firmware_headers(online)
        return

    # Phase 0a: load cached winning datasets if present, else collect.
    # If BOTH bc_dataset_v5.npz (scripted) AND bc_dataset_human.npz
    # (gameplay) exist, they are concatenated — human wins are usually
    # higher quality on hard opponents (rammer/wedger/novamax) where
    # scripted policies orbit-deadlock; combining them gives the DQN
    # the strongest demonstration set.
    sources = []
    if BC_DATASET_PATH.exists():
        sources.append(BC_DATASET_PATH)
    if BC_HUMAN_DATASET_PATH.exists():
        sources.append(BC_HUMAN_DATASET_PATH)

    obs_parts, act_parts, rew_parts = [], [], []
    next_parts, done_parts = [], []
    if sources:
        print(
            f"\nPhase 0a: loading cached dataset(s) "
            f"{[s.name for s in sources]}",
            flush=True,
        )
        for s in sources:
            cached = np.load(s)
            width = cached["obs"].shape[1] if cached["obs"].ndim == 2 else 0
            if width != NET_OBS_DIM:
                print(
                    f"  {s.name}: SKIPPED (obs_dim={width}, need "
                    f"{NET_OBS_DIM}) — stale layout",
                    flush=True,
                )
                continue
            obs_parts.append(cached["obs"])
            act_parts.append(cached["act"])
            rew_parts.append(cached["rew"])
            next_parts.append(cached["next"])
            done_parts.append(cached["done"])
            print(
                f"  {s.name}: {cached['obs'].shape[0]} transitions",
                flush=True,
            )

    if obs_parts:
        obs_arr = np.concatenate(obs_parts, axis=0)
        act_arr = np.concatenate(act_parts, axis=0)
        rew_arr = np.concatenate(rew_parts, axis=0)
        next_arr = np.concatenate(next_parts, axis=0)
        done_arr = np.concatenate(done_parts, axis=0)
        print(
            f"  combined: {obs_arr.shape[0]} transitions "
            f"(obs_dim={obs_arr.shape[1]})",
            flush=True,
        )
    else:
        print(
            f"\nPhase 0a: collecting winning trajectories via "
            f"CombatPolicy (21-D stacked, eps-coverage). Program:",
            flush=True,
        )
        for opp, mult, max_steps, n_pairs in OFFLINE_COLLECTION_PROGRAM:
            print(
                f"  {opp:10s} mult={mult}  max_steps={max_steps}  "
                f"n_pairs_target={n_pairs}",
                flush=True,
            )
        obs_arr, act_arr, rew_arr, next_arr, done_arr = collect_winning_dataset(
            OFFLINE_COLLECTION_PROGRAM,
        )
        np.savez(
            BC_DATASET_PATH,
            obs=obs_arr, act=act_arr, rew=rew_arr,
            next=next_arr, done=done_arr,
        )
        print(
            f"  cached {obs_arr.shape[0]} transitions to "
            f"{BC_DATASET_PATH.name} ({obs_arr.nbytes // 1024} KB obs)",
            flush=True,
        )

    print(
        f"\nPhase 0b: offline pretrain ({OFFLINE_EPOCHS} epochs, "
        f"batch={BATCH_SIZE}, lr={OFFLINE_LR}, gamma={GAMMA})",
        flush=True,
    )
    offline_pretrain(
        online, target,
        obs_arr, act_arr, rew_arr, next_arr, done_arr,
    )
    # Hard-copy target after offline so online fine-tune starts from a
    # consistent point (Polyak τ during offline already nearly synced
    # them; this just removes any residual drift).
    target.load_state_dict(online.state_dict())

    initial_mult, _, initial_opp = ONLINE_TIMESTEPS_PER_PHASE[0]
    print(
        f"\nPhase 1+: online fine-tune, curriculum starts at "
        f"mult={initial_mult}, opp={initial_opp}",
        flush=True,
    )
    stats = online_finetune(
        online, target, initial_mult, initial_opp,
        demos=(obs_arr, act_arr, rew_arr, next_arr, done_arr),
    )
    print(
        f"\nDQN run complete: {stats['total_episodes']} episodes, "
        f"best score {stats['best_score']:.3f}",
        flush=True,
    )

    # Export the DQN advantage head as a deployable C++ header.
    # argmax(Q) == argmax(A) under the dueling decomposition, so the
    # firmware only needs the advantage subnetwork — V head is dropped.
    if BEST_PATH.exists():
        online.load_state_dict(
            torch.load(str(BEST_PATH), map_location="cpu", weights_only=True)
        )
        print(f"Exporting BEST snapshot from {BEST_PATH.name}.", flush=True)
    emit_firmware_headers(online)


def _emit_dqn_header(qnet: DuelingQNet, path: Path) -> None:
    """Render a self-contained C++ header for the dueling DQN's
    advantage head + action lookup table. Output: a single function
    ``predict_action(in_obs[INPUT_DIM]) -> uint8_t`` that runs the
    forward pass through the trunk + advantage head and returns the
    argmax. The host also provides a motor lookup table so firmware
    can map action_idx -> (left_motor, right_motor).
    """
    linears = [m for m in qnet.trunk if isinstance(m, nn.Linear)]
    if len(linears) != 2:
        raise RuntimeError(
            f"expected 2 Linear layers in DQN trunk, got {len(linears)}"
        )
    L1, L2 = linears
    LADV = qnet.advantage_head

    W1 = L1.weight.detach().cpu().numpy().astype(np.float32)
    B1 = L1.bias.detach().cpu().numpy().astype(np.float32)
    W2 = L2.weight.detach().cpu().numpy().astype(np.float32)
    B2 = L2.bias.detach().cpu().numpy().astype(np.float32)
    W_ADV = LADV.weight.detach().cpu().numpy().astype(np.float32)
    B_ADV = LADV.bias.detach().cpu().numpy().astype(np.float32)

    h1, in_dim = W1.shape
    h2, _ = W2.shape
    out_dim, _ = W_ADV.shape

    # Legend indices for the stacked-obs comment block.
    in_dim_k = (in_dim - ENGINEERED_DIM) // 3
    dist_hi = (in_dim - ENGINEERED_DIM) - 1
    eng0 = in_dim - ENGINEERED_DIM
    eng1, eng2, eng3, eng4 = eng0 + 1, eng0 + 2, eng0 + 3, eng0 + 4
    eng5, eng6, eng7, eng8 = eng0 + 5, eng0 + 6, eng0 + 7, eng0 + 8

    def fmt_f(v: float) -> str:
        return f"{v:.7e}f"

    def matrix_lines(mat: np.ndarray) -> str:
        return "\n".join(
            "    {" + ", ".join(fmt_f(v) for v in row) + "},"
            for row in mat
        )

    def vector_line(vec: np.ndarray) -> str:
        return "    " + ", ".join(fmt_f(v) for v in vec)

    # Action lookup table — Cartesian product (left, right) in {-1,0,+1}.
    action_lines = "\n".join(
        f"    {{ {l:+.1f}f, {r:+.1f}f }},"
        for (l, r) in DISCRETE_ACTION_MAP
    )

    # 10 self-test (obs, action_idx) pairs for firmware parity. The input
    # is the K-frame raw-distance stack: the leading (in_dim - 9) columns
    # are stacked distances in [0, 1]; the trailing 9 are the engineered
    # features with their native ranges. Generated for any K = (in_dim-9)/3.
    rng = np.random.default_rng(4242)
    n_dist = in_dim - ENGINEERED_DIM            # 3*K distance columns
    e = n_dist                                  # start of engineered block
    test_obs = np.zeros((10, in_dim), dtype=np.float32)
    test_obs[:, :n_dist] = rng.uniform(0.0, 1.0, size=(10, n_dist))
    test_obs[:, e + 0] = rng.choice([-1.0, 0.0, 1.0], size=10)   # last_seen
    test_obs[:, e + 1:e + 3] = rng.choice([0.0, 1.0], size=(10, 2))  # line_l/r
    test_obs[:, e + 3:e + 5] = rng.uniform(-1.0, 1.0, size=(10, 2))  # prev_l/r
    test_obs[:, e + 5] = rng.uniform(0.0, 1.0, size=10)         # engagement
    test_obs[:, e + 6] = rng.uniform(-1.0, 1.0, size=10)        # yaw_rate
    test_obs[:, e + 7:e + 9] = rng.uniform(-1.0, 1.0, size=(10, 2))  # deltas

    # Compute expected action_idx using a numpy mirror of the AVR fwd.
    def _np_forward(o: np.ndarray) -> int:
        h = np.maximum(0.0, o @ W1.T + B1)
        h = np.maximum(0.0, h @ W2.T + B2)
        a = h @ W_ADV.T + B_ADV
        return int(np.argmax(a))

    test_actions = np.array(
        [_np_forward(test_obs[k]) for k in range(10)], dtype=np.uint8,
    )
    test_obs_lines = "\n".join(
        "    {" + ", ".join(fmt_f(v) for v in test_obs[k]) + "},"
        for k in range(10)
    )
    test_act_lines = "    " + ", ".join(str(int(a)) for a in test_actions)

    header = f"""// Auto-generated by train_dqn.py — do not edit by hand.
//
// Dueling Double DQN, advantage head only (argmax over Q is invariant
// to the value head + mean shift, so AVR firmware can skip V).
//
// Architecture:
//   input({in_dim})
//     -> Linear({h1}) ReLU
//     -> Linear({h2}) ReLU
//     -> Linear({out_dim})    (advantage head, raw logits)
//   action_idx = argmax(advantage)
//   (left_motor, right_motor) = ACTION_MAP[action_idx]
//
// Inputs = K-frame raw-distance stack (oldest distances first) followed
// by the 9 single-frame engineered features. For INPUT_DIM={in_dim}, K={in_dim_k}:
//   [0 .. {dist_hi}]  stacked (front,left,right) x K, each in [0, 1]
//                 layout: f[t-K+1],l,r, ... , f[t],l[t],r[t]
//   [{eng0}] last_seen        in {{-1, 0, +1}}
//   [{eng1}] line_l           in {{0, 1}}
//   [{eng2}] line_r           in {{0, 1}}
//   [{eng3}] prev_left        in [-1, +1]
//   [{eng4}] prev_right       in [-1, +1]
//   [{eng5}] engagement       in [0, 1]
//   [{eng6}] yaw_rate_proxy   in [-1, +1]
//   [{eng7}] front_ir_delta   in [-1, +1]
//   [{eng8}] lateral_ir_delta in [-1, +1]

#pragma once
#include <avr/pgmspace.h>
#include <stdint.h>

namespace mini_sumo_ai {{

constexpr uint8_t INPUT_DIM  = {in_dim};
constexpr uint8_t H1_DIM     = {h1};
constexpr uint8_t H2_DIM     = {h2};
constexpr uint8_t N_ACTIONS  = {out_dim};

// === Weights =========================================================
const float W1[{h1}][{in_dim}] PROGMEM = {{
{matrix_lines(W1)}
}};
const float B1[{h1}] PROGMEM = {{
{vector_line(B1)}
}};

const float W2[{h2}][{h1}] PROGMEM = {{
{matrix_lines(W2)}
}};
const float B2[{h2}] PROGMEM = {{
{vector_line(B2)}
}};

const float W_ADV[{out_dim}][{h2}] PROGMEM = {{
{matrix_lines(W_ADV)}
}};
const float B_ADV[{out_dim}] PROGMEM = {{
{vector_line(B_ADV)}
}};

// === Action lookup table ============================================
// idx -> (left_motor, right_motor) in [-1, +1].
const float ACTION_MAP[N_ACTIONS][2] PROGMEM = {{
{action_lines}
}};

inline float relu(float x) {{ return x > 0.0f ? x : 0.0f; }}

inline uint8_t predict_action(const float in[INPUT_DIM]) {{
    float h1_act[H1_DIM];
    for (uint8_t i = 0; i < H1_DIM; ++i) {{
        float s = pgm_read_float(&B1[i]);
        for (uint8_t j = 0; j < INPUT_DIM; ++j) {{
            s += pgm_read_float(&W1[i][j]) * in[j];
        }}
        h1_act[i] = relu(s);
    }}
    float h2_act[H2_DIM];
    for (uint8_t i = 0; i < H2_DIM; ++i) {{
        float s = pgm_read_float(&B2[i]);
        for (uint8_t j = 0; j < H1_DIM; ++j) {{
            s += pgm_read_float(&W2[i][j]) * h1_act[j];
        }}
        h2_act[i] = relu(s);
    }}
    // Advantage head, argmax. No mean subtraction needed: it shifts
    // all advantages by the same constant so doesn't affect argmax.
    float best = -1e30f;
    uint8_t best_idx = 0;
    for (uint8_t a = 0; a < N_ACTIONS; ++a) {{
        float s = pgm_read_float(&B_ADV[a]);
        for (uint8_t j = 0; j < H2_DIM; ++j) {{
            s += pgm_read_float(&W_ADV[a][j]) * h2_act[j];
        }}
        if (s > best) {{ best = s; best_idx = a; }}
    }}
    return best_idx;
}}

inline void action_to_motors(uint8_t idx, float &left, float &right) {{
    left  = pgm_read_float(&ACTION_MAP[idx][0]);
    right = pgm_read_float(&ACTION_MAP[idx][1]);
}}

// === Self-test ========================================================
// 10 (obs, expected_action_idx) pairs computed in Python with the
// same forward pass. A healthy port should return 0 mismatches.

constexpr uint8_t SELFTEST_N = 10;
const float SELFTEST_INPUTS[SELFTEST_N][INPUT_DIM] PROGMEM = {{
{test_obs_lines}
}};
const uint8_t SELFTEST_EXPECTED[SELFTEST_N] PROGMEM = {{
{test_act_lines}
}};

inline uint8_t verify_self_test() {{
    uint8_t mismatches = 0;
    for (uint8_t k = 0; k < SELFTEST_N; ++k) {{
        float in[INPUT_DIM];
        for (uint8_t j = 0; j < INPUT_DIM; ++j) {{
            in[j] = pgm_read_float(&SELFTEST_INPUTS[k][j]);
        }}
        uint8_t got = predict_action(in);
        uint8_t expected = pgm_read_byte(&SELFTEST_EXPECTED[k]);
        if (got != expected) ++mismatches;
    }}
    return mismatches;
}}

}}  // namespace mini_sumo_ai
"""
    # UTF-8 because the header includes em-dashes and other Unicode
    # punctuation in comments. AVR toolchain handles UTF-8 fine.
    path.write_text(header, encoding="utf-8")


def emit_firmware_headers(qnet: DuelingQNet) -> None:
    """Write the C++ header to BOTH the canonical firmware/ copy and the
    firmware/v3_deploy/ copy the deployed sketch #includes. They were
    previously hand-synced; emitting both keeps them byte-identical."""
    targets = [
        FIRMWARE / "neural_net_v6_3d.h",
        FIRMWARE / "v3_deploy" / "neural_net_v6_3d.h",
    ]
    for t in targets:
        if not t.parent.exists():
            print(f"  skip {t} (dir missing)", flush=True)
            continue
        _emit_dqn_header(qnet, t)
        print(f"Wrote {t.relative_to(HERE_TOP)}", flush=True)


if __name__ == "__main__":
    main()
