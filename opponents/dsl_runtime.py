"""Safe interpreter for a user-authored :class:`OpponentDSL`.

``DslOpponent`` is a drop-in :class:`OpponentController`: it composes the same
``EdgeRecovery`` as every built-in (so edge survival always wins), then walks
the validated DSL rules and returns the FIRST matching action's wheel command,
clamped to the env's caps. If nothing matches it runs the program's default
action.

Interpretation is PURE — predicates resolve to boolean reads of the ``decide``
inputs and combinators (``all``/``any``/``not``/``timer``) recurse over the
already-validated tree. There is no ``eval``/``exec`` and no dynamic attribute
lookup: an unknown predicate would have been rejected by ``validate`` at
construction time, and the dispatch table here is a fixed dict of lambdas over
the inputs only.
"""

from __future__ import annotations

from typing import Any

from webapp.shared.opponent_dsl import ACTION_WHEELS, DSL_SPEED, OpponentDSL

from .base import OpponentController

# Clamp magnitude for the DSL's wheel outputs (rad/s). The env additionally
# clips to the opponent's physical caps via ``_novamax_caps``; this is a
# defensive ceiling so a runtime command can never exceed the action table.
_CAP = DSL_SPEED


def _clamp(v: float) -> float:
    return max(-_CAP, min(_CAP, float(v)))


class DslOpponent(OpponentController):
    """Interpret a validated :class:`OpponentDSL` as a sumo opponent."""

    def __init__(self, dsl: OpponentDSL | dict | list) -> None:
        super().__init__()
        # Accept a parsed DSL or a raw dict/list (validated on parse).
        self.dsl = dsl if isinstance(dsl, OpponentDSL) else OpponentDSL.from_dict(dsl)
        self._tick = 0

    def reset(self) -> None:
        super().reset()
        self._tick = 0

    # -- condition evaluation (pure, table-driven) ------------------------
    def _predicate(self, name: str, ir: dict[str, bool], el: bool, er: bool) -> bool:
        """Resolve a single predicate name against the decide inputs."""
        if name == "front_hit":
            return bool(ir.get("fc"))
        if name == "left_hit":
            return bool(ir.get("fl"))
        if name == "right_hit":
            return bool(ir.get("fr"))
        if name == "side_left_hit":
            return bool(ir.get("sl"))
        if name == "side_right_hit":
            return bool(ir.get("sr"))
        if name == "edge_left":
            return bool(el)
        if name == "edge_right":
            return bool(er)
        if name == "no_target":
            return not any(ir.get(k) for k in ("fc", "fl", "fr", "sl", "sr"))
        # Unreachable: validate() rejects unknown predicates before construction.
        raise ValueError(f"unknown predicate at runtime: {name!r}")

    def _eval(self, cond: Any, ir: dict[str, bool], el: bool, er: bool) -> bool:
        """Recursively evaluate a (validated) condition node to a bool."""
        if isinstance(cond, str):
            return self._predicate(cond, ir, el, er)
        # validate() guarantees a single-key dict with a known combinator.
        (key, val), = cond.items()
        if key == "all":
            return all(self._eval(c, ir, el, er) for c in val)
        if key == "any":
            return any(self._eval(c, ir, el, er) for c in val)
        if key == "not":
            return not self._eval(val, ir, el, er)
        if key == "timer":
            # Fires on every Nth tick (1-indexed): true when tick % N == 0.
            return (self._tick % int(val["every"])) == 0
        raise ValueError(f"unknown combinator at runtime: {key!r}")

    # -- controller surface ----------------------------------------------
    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        # Priority 1: edge survival (shared zoo behaviour + env watchdog).
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge

        self._tick += 1
        for rule in self.dsl.rules:
            if self._eval(rule.when, ir_hits, edge_left, edge_right):
                l, r = ACTION_WHEELS[rule.do]
                return _clamp(l), _clamp(r)

        l, r = ACTION_WHEELS[self.dsl.default]
        return _clamp(l), _clamp(r)
