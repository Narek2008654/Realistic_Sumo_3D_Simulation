"""SAFE, data-driven behavior DSL for user-authored sumo opponents.

A behavior is an ORDERED list of rules ``[{"when": <cond>, "do": <action>}]``.
At each control tick the runtime walks the rules top-to-bottom and fires the
FIRST rule whose condition matches; its action maps to a bounded ``(l, r)``
wheel command in rad/s. If no rule matches a configurable default action runs.

There is **no eval/exec anywhere**: a condition is a nested dict over a FIXED
vocabulary of predicates and the boolean combinators ``all`` / ``any`` /
``not``, plus an optional ``timer`` ({"every": N}). An action is one of a
FIXED set of names. Anything outside the vocabulary is rejected by
:func:`validate` before it ever reaches the runtime.

Pure Python-3.12 stdlib (``dataclasses``) — no third-party deps — so it can be
imported by the core ``opponents`` package without dragging in the web stack.

Inputs available to conditions come ONLY from the controller ``decide``
signature ``(ir_hits, edge_left, edge_right)`` where ``ir_hits`` has keys
``fc, fl, fr, sl, sr``:

    Predicate            True when
    -----------------    ----------------------------------------------
    front_hit            ir_hits["fc"]  (front-center beam sees a target)
    left_hit             ir_hits["fl"]  (front-left beam)
    right_hit            ir_hits["fr"]  (front-right beam)
    side_left_hit        ir_hits["sl"]  (side-left beam)
    side_right_hit       ir_hits["sr"]  (side-right beam)
    edge_left            edge_left  (left line sensor over the border)
    edge_right           edge_right (right line sensor over the border)
    no_target            none of fc/fl/fr/sl/sr is set

Actions (bounded wheel commands, rad/s — the env additionally clamps these to
the opponent's physical caps):

    forward, reverse, spin_left, spin_right, arc_left, arc_right, stop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PREDICATES",
    "ACTIONS",
    "ACTION_WHEELS",
    "DSL_SPEED",
    "Rule",
    "OpponentDSL",
    "validate",
]

# Speed magnitude used by the action table (rad/s). Matches the zoo's FULL
# (83.78 ≈ 800 RPM); the env clips this down to the opponent's actual caps via
# ``_novamax_caps`` exactly as it does for the built-in controllers.
DSL_SPEED: float = 83.78

# Fixed predicate vocabulary. Each maps to a pure boolean over the decide
# inputs; the runtime resolves these — they are NOT code, just names.
PREDICATES: frozenset[str] = frozenset(
    {
        "front_hit",
        "left_hit",
        "right_hit",
        "side_left_hit",
        "side_right_hit",
        "edge_left",
        "edge_right",
        "no_target",
    }
)

# Boolean combinators allowed as condition keys.
_COMBINATORS: frozenset[str] = frozenset({"all", "any", "not", "timer"})

# Fixed action vocabulary -> bounded (l, r) in rad/s.
_S = DSL_SPEED
ACTION_WHEELS: dict[str, tuple[float, float]] = {
    "forward": (_S, _S),
    "reverse": (-_S, -_S),
    "spin_left": (-_S, _S),
    "spin_right": (_S, -_S),
    "arc_left": (_S * 0.4, _S),
    "arc_right": (_S, _S * 0.4),
    "stop": (0.0, 0.0),
}
ACTIONS: frozenset[str] = frozenset(ACTION_WHEELS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_cond(cond: Any, path: str, errors: list[str]) -> None:
    """Recursively validate a condition node, appending human errors."""
    if isinstance(cond, str):
        if cond not in PREDICATES:
            errors.append(f"{path}: unknown predicate {cond!r}")
        return
    if not isinstance(cond, dict):
        errors.append(
            f"{path}: condition must be a predicate name or a dict, "
            f"got {type(cond).__name__}"
        )
        return
    if len(cond) != 1:
        errors.append(
            f"{path}: condition dict must have exactly one key "
            f"(one of {sorted(_COMBINATORS)}), got keys {sorted(cond)}"
        )
        return
    (key, val), = cond.items()
    if key not in _COMBINATORS:
        errors.append(
            f"{path}: unknown combinator {key!r} "
            f"(allowed: {sorted(_COMBINATORS)})"
        )
        return
    if key in ("all", "any"):
        if not isinstance(val, list) or not val:
            errors.append(f"{path}.{key}: must be a non-empty list of conditions")
            return
        for i, child in enumerate(val):
            _validate_cond(child, f"{path}.{key}[{i}]", errors)
    elif key == "not":
        _validate_cond(val, f"{path}.not", errors)
    elif key == "timer":
        if not isinstance(val, dict) or set(val) != {"every"}:
            errors.append(f"{path}.timer: must be {{'every': N}} with integer N>=1")
            return
        n = val["every"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            errors.append(f"{path}.timer.every: must be an integer >= 1, got {n!r}")


def validate(dsl: Any) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid).

    Accepts either a raw dict ``{"rules": [...], "default": <action>}`` or a
    bare list of rules. Rejects unknown predicates, actions, combinators, and
    stray keys.
    """
    errors: list[str] = []

    if isinstance(dsl, list):
        rules, default = dsl, "stop"
    elif isinstance(dsl, dict):
        extra = set(dsl) - {"rules", "default"}
        if extra:
            errors.append(f"unknown top-level keys: {sorted(extra)}")
        rules = dsl.get("rules", [])
        default = dsl.get("default", "stop")
    else:
        return [f"behavior_dsl must be a dict or list, got {type(dsl).__name__}"]

    if not isinstance(default, str) or default not in ACTIONS:
        errors.append(
            f"default: must be one of {sorted(ACTIONS)}, got {default!r}"
        )

    if not isinstance(rules, list):
        errors.append(f"rules: must be a list, got {type(rules).__name__}")
        return errors
    if not rules:
        errors.append("rules: must contain at least one rule")

    for i, rule in enumerate(rules):
        rp = f"rules[{i}]"
        if not isinstance(rule, dict):
            errors.append(f"{rp}: each rule must be a dict, got {type(rule).__name__}")
            continue
        extra = set(rule) - {"when", "do"}
        if extra:
            errors.append(f"{rp}: unknown rule keys: {sorted(extra)}")
        if "when" not in rule:
            errors.append(f"{rp}: missing 'when'")
        else:
            _validate_cond(rule["when"], f"{rp}.when", errors)
        do = rule.get("do")
        if not isinstance(do, str) or do not in ACTIONS:
            errors.append(f"{rp}.do: must be one of {sorted(ACTIONS)}, got {do!r}")

    return errors


# ---------------------------------------------------------------------------
# Parsed representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """One ``{when, do}`` rule. ``when`` is the raw (validated) condition node."""

    when: Any
    do: str


@dataclass(frozen=True)
class OpponentDSL:
    """A validated, ordered opponent behavior program."""

    rules: tuple[Rule, ...]
    default: str = "stop"

    @classmethod
    def from_dict(cls, data: Any) -> "OpponentDSL":
        """Build a validated DSL from a dict/list. Raises ValueError if invalid."""
        errors = validate(data)
        if errors:
            raise ValueError("invalid behavior_dsl: " + "; ".join(errors))
        if isinstance(data, list):
            rules_raw, default = data, "stop"
        else:
            rules_raw = data.get("rules", [])
            default = data.get("default", "stop")
        rules = tuple(Rule(when=r["when"], do=r["do"]) for r in rules_raw)
        return cls(rules=rules, default=default)

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to the canonical ``{rules, default}`` dict."""
        return {
            "rules": [{"when": r.when, "do": r.do} for r in self.rules],
            "default": self.default,
        }
