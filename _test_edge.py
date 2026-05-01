"""Stress test: park NovaMax just past the inner ring and watch edge recovery."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401
import numpy as np
import pybullet as p
from sumo_env import MiniSumoEnv, INNER_RADIUS, DOHYO_RADIUS

env = MiniSumoEnv(gui=False, seed=0, novamax_level=3)
env.reset()

ep, eo = p.getBasePositionAndOrientation(env.enemy_id)
p.resetBasePositionAndOrientation(env.enemy_id, [0.31, 0.0, ep[2]], eo)
print(f"Placed enemy at radius 0.31  (inner={INNER_RADIUS:.3f}, outer={DOHYO_RADIUS:.3f})")

for i in range(15):
    obs, r, t, tr, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    ep, _ = p.getBasePositionAndOrientation(env.enemy_id)
    rad = (ep[0]**2 + ep[1]**2) ** 0.5
    on_dohyo = ep[2] >= 0.0
    print(f"  step {i:2d}  radius={rad:.3f}  z={ep[2]:.4f}  "
          f"braking={env._enemy_ctrl.is_edge_braking}  on_dohyo={on_dohyo}")
    if t or tr:
        print("  episode ended"); break
env.close()
