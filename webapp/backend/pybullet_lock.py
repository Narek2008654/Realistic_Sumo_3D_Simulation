"""Process-wide lock serializing in-process PyBullet use.

The sim env calls ``p.*`` against PyBullet's default (most-recently-connected)
client, so two PyBullet envs alive at once in a single process corrupt each
other and 500 the request (e.g. clicking *Evaluate* on two models at once).
FastAPI runs sync endpoints in a threadpool, so every handler that builds a
PyBullet env in-process (model evaluation, URDF validation) must hold this lock;
concurrent callers then queue and run one at a time instead of crashing.

Out-of-process work (the training / eval-recording subprocesses) has its own
client and is unaffected.
"""

import threading

pybullet_lock = threading.Lock()
