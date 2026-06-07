"""Regression gate for the spec-driven observation refactor (feature E1b).

The env's distance sensing and observation assembly are now driven by a
``HardwareSpec`` (``HardwareSpec.default()`` for the default robot). This test
re-runs the exact deterministic rollout captured in
``tests/golden/_capture_obs.py`` and asserts the refactored env reproduces
the pre-refactor golden observation stream byte-for-byte (max abs diff < 1e-9)
for BOTH the stacked observation and the raw distance channels.

Run directly:
    python tests/test_obs_faithfulness_spec.py
or under pytest:
    pytest tests/test_obs_faithfulness_spec.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch FIRST on Windows (DLL-search-order race vs numpy); the rollout
# pulls in train_dqn_3d, which requires torch.
import torch  # noqa: F401

import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.path.insert(0, os.path.join(_HERE, "golden"))
from _capture_obs import rollout, OBS_GOLDEN, RAW_GOLDEN  # noqa: E402

TOL = 1e-9


def test_obs_faithfulness_spec() -> None:
    """Refactored env == golden master, to < 1e-9, for obs and raw dists."""
    assert os.path.exists(OBS_GOLDEN), (
        f"missing golden {OBS_GOLDEN}; regenerate with "
        f"`python tests/golden/_capture_obs.py` on a known-good revision"
    )
    assert os.path.exists(RAW_GOLDEN), f"missing golden {RAW_GOLDEN}"

    golden_obs = np.load(OBS_GOLDEN)
    golden_raw = np.load(RAW_GOLDEN)

    new_obs, new_raw = rollout()

    assert new_obs.shape == golden_obs.shape, (
        f"stacked obs shape {new_obs.shape} != golden {golden_obs.shape}"
    )
    assert new_raw.shape == golden_raw.shape, (
        f"raw shape {new_raw.shape} != golden {golden_raw.shape}"
    )

    obs_diff = float(np.max(np.abs(new_obs - golden_obs)))
    raw_diff = float(np.max(np.abs(new_raw - golden_raw)))

    assert obs_diff < TOL, f"stacked obs max abs diff {obs_diff:.3e} >= {TOL:.0e}"
    assert raw_diff < TOL, f"raw distance max abs diff {raw_diff:.3e} >= {TOL:.0e}"


if __name__ == "__main__":
    test_obs_faithfulness_spec()
    golden_obs = np.load(OBS_GOLDEN)
    golden_raw = np.load(RAW_GOLDEN)
    new_obs, new_raw = rollout()
    print(
        "PASS  stacked obs max|diff| =",
        f"{float(np.max(np.abs(new_obs - golden_obs))):.3e}",
        " raw max|diff| =",
        f"{float(np.max(np.abs(new_raw - golden_raw))):.3e}",
    )
