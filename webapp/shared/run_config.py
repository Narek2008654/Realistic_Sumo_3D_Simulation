"""Training job configuration contract (E1f).

A ``TrainingConfig`` is the JSON seam between a job launcher (the web UI or a
CLI) and the trainers. It is read ONLY when the env var ``SUMO_RUN_CONFIG``
points at a JSON file; when unset, the trainers ignore this module entirely
and behave byte-identically to their hard-coded defaults.

Pure Python-3.12 stdlib (``dataclasses``, ``json``) plus
:class:`HardwareSpec` (itself stdlib-only) — no third-party deps — so it is
safe to import from the core trainers without dragging in the web stack.

JSON schema (all keys optional except ``algo`` and ``total_steps``)::

    {
      "algo": "dqn" | "ppo",
      "total_steps": 12000,
      "eval_every": 2000,
      "output_best_path": "<path>.pt",
      "output_final_path": "<path>.pt",
      "job_dir": "<dir>",
      "opponent_weights": {"novamax": 1.0, ...} | null,
      "resume_path": "<path>.pt" | null,
      "hardware_spec": {<HardwareSpec.to_dict()>} | null,
      "seed": 42
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .hardware_spec import HardwareSpec


@dataclass
class TrainingConfig:
    """A single training job's configuration.

    ``job_dir`` is the root the periodic hook writes under (snapshots/,
    trajectories/, progress.jsonl). ``eval_every`` is the cadence (in env
    steps) at which the checkpoint/eval/trajectory hook fires.
    """

    algo: str
    total_steps: int
    eval_every: int = 100_000
    output_best_path: Optional[str] = None
    output_final_path: Optional[str] = None
    job_dir: Optional[str] = None
    opponent_weights: Optional[dict] = None
    resume_path: Optional[str] = None
    hardware_spec: Optional[HardwareSpec] = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-ready dict.

        ``hardware_spec`` is emitted via :meth:`HardwareSpec.to_dict` (or
        ``None``); every other field is already JSON-native.
        """
        d = asdict(self)
        # asdict() turns the nested HardwareSpec dataclass into a dict, but
        # only if it is a dataclass instance. Normalise explicitly so a None
        # stays None and a spec round-trips through HardwareSpec.to_dict.
        if self.hardware_spec is None:
            d["hardware_spec"] = None
        else:
            d["hardware_spec"] = self.hardware_spec.to_dict()
        return d

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def load(path: str | Path) -> TrainingConfig:
    """Read a ``TrainingConfig`` from a JSON file.

    ``hardware_spec``, if present and non-null, is rebuilt via
    :meth:`HardwareSpec.from_dict`. Missing optional keys fall back to the
    dataclass defaults. ``algo`` and ``total_steps`` are required.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)

    if "algo" not in data:
        raise ValueError(f"run config {path!s}: missing required key 'algo'")
    if "total_steps" not in data:
        raise ValueError(
            f"run config {path!s}: missing required key 'total_steps'"
        )

    hw = data.get("hardware_spec")
    hardware_spec = HardwareSpec.from_dict(hw) if hw else None

    return TrainingConfig(
        algo=str(data["algo"]),
        total_steps=int(data["total_steps"]),
        eval_every=int(data.get("eval_every", 100_000)),
        output_best_path=data.get("output_best_path"),
        output_final_path=data.get("output_final_path"),
        job_dir=data.get("job_dir"),
        opponent_weights=data.get("opponent_weights"),
        resume_path=data.get("resume_path"),
        hardware_spec=hardware_spec,
        seed=int(data.get("seed", 42)),
    )
