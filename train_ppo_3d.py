"""Discrete PPO for the mini-sumo agent (branch pol/ppo).

A drop-in algorithm swap: same 21-D RawDistanceStack env, same discrete
9-action grid, same reward shaping (flank + base + hardcoded safety
override) as the DQN line — PPO instead of Dueling-Double-DQN.

Design notes:
- The ActorCritic deliberately reuses DuelingQNet's submodule NAMES
  (`trunk`, `advantage_head`, `value_head`) so every existing tool works
  on a PPO checkpoint unchanged: eval_best / watch_3d infer the arch and
  call `act_greedy` (argmax of `advantage_head` = the action), and
  `_emit_dqn_header` exports the actor head for the Nano (critic dropped,
  exactly like the DQN V-head). So a trained PPO policy is deployable via
  firmware/v4_deploy with zero export changes.
- Anti-forgetting (the critics' fixes): BC-initialize the ACTOR ONLY from
  the winner demos (never the critic), normalize advantages, sample the
  full zoo every episode (no single-opponent phase), and ramp only the
  physics mult.
- Single PyBullet env (the env calls p.* without physicsClientId, so it is
  not safe to run several envs in one process); subprocess vec-envs are a
  later throughput upgrade. Env stepping dominates wall-clock, ~as DQN.

Run:   cmd /c "call ...activate.bat sumo && set KMP_DUPLICATE_LIB_OK=TRUE && python -u train_ppo_3d.py"
Smoke: set PPO_SMOKE=1 first (tiny run to validate the loop end-to-end).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import numpy as np
import random
import time
from collections import deque
from torch.distributions import Categorical

from train_dqn_3d import (
    build_env, NET_OBS_DIM, N_ACTIONS, CHECKPOINTS, BC_DATASET_PATH,
    save_checkpoint_atomic,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
NET_ARCH = (32, 32)            # Nano-deployable, matches the DQN line
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
ENT_COEF = 0.01                # entropy bonus (keeps exploration up early)
VF_COEF = 0.5
LR = 2.5e-4
EPOCHS = 4                     # PPO epochs per rollout
MINIBATCH = 256
ROLLOUT = 2048                 # env steps per policy update
TOTAL_STEPS = 1_000_000
MAX_GRAD_NORM = 0.5
BC_EPOCHS = 10                 # actor-only behavior-cloning warm start
LOG_EVERY = 5_000

# Curriculum: ramp only the physics mult; full zoo every episode throughout.
MULT_PHASES = ((0.7, 100_000), (1.0, 200_000), (1.5, 200_000),
               (2.0, 200_000), (3.0, 300_000))

BEST_PATH = CHECKPOINTS / "ppo_stack_best.pt"
FINAL_PATH = CHECKPOINTS / "ppo_stack_final.pt"

SMOKE = bool(os.environ.get("PPO_SMOKE"))
if SMOKE:
    ROLLOUT = 512
    TOTAL_STEPS = 8_000
    BC_EPOCHS = 1
    MULT_PHASES = ((1.0, 4_000), (1.5, 4_000))
    LOG_EVERY = 1_000


# ---------------------------------------------------------------------------
# Actor-critic (export-compatible with DuelingQNet tooling)
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=NET_ARCH):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.advantage_head = nn.Linear(prev, n_actions)   # actor logits
        self.value_head = nn.Linear(prev, 1)               # critic
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.advantage_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        """Deterministic argmax of the actor head — used by eval/watch and
        what the exported firmware computes."""
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.advantage_head(self.trunk(x))
        return int(torch.argmax(logits, dim=1).item())

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


# ---------------------------------------------------------------------------
# Behavior cloning: actor-only warm start from winner demos
# ---------------------------------------------------------------------------
def bc_init_actor(net: ActorCritic, obs_arr, act_arr, epochs=BC_EPOCHS,
                  batch=256, lr=1e-3):
    """Cross-entropy on the demonstrated (obs -> action) pairs, ACTOR ONLY.
    The critic is left at init (it warms up on-policy); BC'ing a critic on
    winner-only returns would bias V high and corrupt early advantages."""
    if obs_arr.shape[0] == 0:
        print("bc_init skipped: empty demo set", flush=True)
        return
    params = list(net.trunk.parameters()) + list(net.advantage_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
    act_t = torch.as_tensor(act_arr, dtype=torch.long)
    n = obs_arr.shape[0]
    for ep in range(epochs):
        idx = np.random.permutation(n)
        losses = []
        for s in range(0, n, batch):
            sel = idx[s:s + batch]
            logits = net.advantage_head(net.trunk(obs_t[sel]))
            loss = nn.functional.cross_entropy(logits, act_t[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        acc = float((net.advantage_head(net.trunk(obs_t)).argmax(1) == act_t).float().mean())
        print(f"  bc actor epoch {ep+1}/{epochs} loss={np.mean(losses):.4f} acc={acc:.2%}",
              flush=True)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """Generalized advantage estimation over one rollout. `dones[t]` marks
    the episode boundary after step t; bootstrapping stops across it."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_v = last_value if t == n - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = build_env(
        gui=False, seed=SEED,
        novamax_torque_mult=MULT_PHASES[0][0], force_opponent_id=None,
        narek_reward=True, action_consistency_reward=True,
        flank_reward=True, safety_override=True,
    )

    net = ActorCritic(NET_OBS_DIM, N_ACTIONS, hidden=NET_ARCH)

    # Actor-only BC warm start from the cached winner demos (21-D).
    if BC_DATASET_PATH.exists():
        cached = np.load(BC_DATASET_PATH)
        if cached["obs"].shape[1] == NET_OBS_DIM:
            print(f"BC warm start from {BC_DATASET_PATH.name} "
                  f"({cached['obs'].shape[0]} demos)", flush=True)
            bc_init_actor(net, cached["obs"], cached["act"])
        else:
            print(f"BC skipped: {BC_DATASET_PATH.name} is "
                  f"{cached['obs'].shape[1]}-D, need {NET_OBS_DIM}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=LR)

    obs, _ = env.reset(seed=SEED)
    step = 0
    episodes = 0
    recent = deque(maxlen=200)
    per_opp: dict[str, deque] = {}
    best_score = -1.0
    mult_idx = 0
    phase_end = MULT_PHASES[0][1]
    env.unwrapped.novamax_torque_mult = float(MULT_PHASES[0][0])
    t0 = time.time()
    last_log = 0

    while step < TOTAL_STEPS:
        # advance the mult curriculum
        if step >= phase_end and mult_idx < len(MULT_PHASES) - 1:
            mult_idx += 1
            env.unwrapped.novamax_torque_mult = float(MULT_PHASES[mult_idx][0])
            phase_end += MULT_PHASES[mult_idx][1]
            recent.clear()
            print(f"\n=== PPO mult phase {mult_idx+1}/{len(MULT_PHASES)}: "
                  f"mult={MULT_PHASES[mult_idx][0]} (step {step}) ===", flush=True)

        # ---- collect a rollout ----
        b_obs, b_act, b_logp, b_rew, b_val, b_done = [], [], [], [], [], []
        for _ in range(ROLLOUT):
            a, logp, val = net.act(obs)
            next_obs, reward, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            b_obs.append(obs); b_act.append(a); b_logp.append(logp)
            b_rew.append(float(reward)); b_val.append(val); b_done.append(float(done))
            obs = next_obs
            step += 1
            if done:
                episodes += 1
                reason = info.get("termination_reason", "")
                win = 1 if reason == "win" else 0
                recent.append(win)
                opp = info.get("opponent_id", "?")
                per_opp.setdefault(opp, deque(maxlen=200)).append(win)
                obs, _ = env.reset()

        with torch.no_grad():
            _, last_value = net.forward(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            last_value = float(last_value.item())

        adv, returns = compute_gae(b_rew, b_val, b_done, last_value)
        # advantage normalization (critics' fix for the 4-OOM reward scale)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(np.asarray(b_obs), dtype=torch.float32)
        act_t = torch.as_tensor(np.asarray(b_act), dtype=torch.long)
        old_logp_t = torch.as_tensor(np.asarray(b_logp), dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        # ---- PPO update ----
        n = len(b_obs)
        for _ in range(EPOCHS):
            perm = np.random.permutation(n)
            for s in range(0, n, MINIBATCH):
                mb = perm[s:s + MINIBATCH]
                new_logp, entropy, value = net.evaluate(obs_t[mb], act_t[mb])
                ratio = torch.exp(new_logp - old_logp_t[mb])
                a_mb = adv_t[mb]
                surr1 = ratio * a_mb
                surr2 = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * a_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, ret_t[mb])
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                opt.step()

        # ---- log + best checkpoint ----
        if step - last_log >= LOG_EVERY:
            last_log = step
            wr = sum(recent) / len(recent) if recent else 0.0
            mult = MULT_PHASES[mult_idx][0]
            score = wr * mult
            marker = ""
            if score > best_score and len(recent) >= 100:
                best_score = score
                save_checkpoint_atomic(net.state_dict(), BEST_PATH)
                marker = "  *BEST*"
            fps = step / max(1e-6, time.time() - t0)
            print(f"step {step:7d}  ep={episodes:5d}  wr({len(recent)})={wr:.2%}  "
                  f"ent={float(entropy.mean()):.3f}  fps={fps:.1f}{marker}", flush=True)
            opp_strs = [f"{o}={sum(v)/max(1,len(v)):.0%}({len(v)})"
                        for o, v in sorted(per_opp.items())]
            if opp_strs:
                print("  per_opp: " + "  ".join(opp_strs), flush=True)

    save_checkpoint_atomic(net.state_dict(), FINAL_PATH)
    print(f"\nPPO done: {episodes} episodes, best score {best_score:.3f}", flush=True)


if __name__ == "__main__":
    train()
