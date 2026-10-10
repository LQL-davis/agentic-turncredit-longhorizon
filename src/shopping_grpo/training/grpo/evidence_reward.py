"""Training-only evidence diagnostics for long-horizon shopping rollouts.

The scorer in this module is deliberately shadow-only.  It summarizes public,
bounded progress counters returned by ShopSimulator without changing Reward v3,
the sequence reward, dynamic-sampling eligibility, or evaluation success.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


SHADOW_REWARD_VERSION = "shopping-evidence-shadow-v2"
EVIDENCE_KEYS = ("constraint", "option", "product", "subpage", "result_set")
DEFAULT_CAPS = {
    "constraint": 8,
    "option": 4,
    "product": 5,
    "subpage": 6,
    "result_set": 3,
}
DEFAULT_WEIGHTS = {
    "constraint": 0.40,
    "option": 0.20,
    "product": 0.15,
    "subpage": 0.15,
    "result_set": 0.10,
}
EVIDENCE_PREFIXES = tuple(f"{key}:" for key in EVIDENCE_KEYS)
MAX_EVIDENCE_KEY_LENGTH = 512


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0 or parsed != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def sanitize_progress_snapshot(raw: object) -> dict:
    """Validate and minimize the public progress fields needed by the scorer."""
    if not isinstance(raw, Mapping):
        raise ValueError("progress snapshot must be an object")
    counts = raw.get("evidence_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("progress.evidence_counts must be an object")
    raw_added = raw.get("credited_evidence_added", raw.get("evidence_added", ()))
    if not isinstance(raw_added, Sequence) or isinstance(raw_added, (str, bytes)):
        raise ValueError("progress.credited_evidence_added must be an array")
    credited_evidence_added = []
    for index, value in enumerate(raw_added):
        if not isinstance(value, str):
            raise ValueError(
                f"progress.credited_evidence_added[{index}] must be a string"
            )
        key = value.strip()
        if (
            not key
            or len(key) > MAX_EVIDENCE_KEY_LENGTH
            or any(ord(character) < 32 for character in key)
            or not key.startswith(EVIDENCE_PREFIXES)
        ):
            raise ValueError(
                f"progress.credited_evidence_added[{index}] is invalid"
            )
        credited_evidence_added.append(key)

    return {
        "step_count": _non_negative_int(raw.get("step_count", 0), "step_count"),
        "consecutive_repeats": _non_negative_int(
            raw.get("consecutive_repeats", 0), "consecutive_repeats"
        ),
        "no_progress_steps": _non_negative_int(
            raw.get("no_progress_steps", 0), "no_progress_steps"
        ),
        "effective_result_sets": _non_negative_int(
            raw.get("effective_result_sets", 0), "effective_result_sets"
        ),
        "opened_candidate_count": _non_negative_int(
            raw.get("opened_candidate_count", 0), "opened_candidate_count"
        ),
        "evidence_counts": {
            key: _non_negative_int(counts.get(key, 0), f"evidence_counts.{key}")
            for key in EVIDENCE_KEYS
        },
        "credited_evidence_added": credited_evidence_added,
    }


def _empty_quality_state() -> dict:
    return {"result_set_seen": False, "candidates": {}, "seen_keys": set()}


def _candidate(state: dict, asin: str) -> dict:
    return state["candidates"].setdefault(
        asin,
        {
            "product": False,
            "hard_gates": {},
            "options": set(),
            "subpages": set(),
        },
    )


def _apply_quality_evidence(state: dict, evidence_keys: Sequence[str]) -> None:
    """Update candidate-local public evidence without using hidden gold data."""
    for key in evidence_keys:
        if key in state["seen_keys"]:
            continue
        state["seen_keys"].add(key)
        parts = key.split(":", 4)
        kind = parts[0]
        if kind == "result_set":
            state["result_set_seen"] = True
        elif kind == "product" and len(parts) >= 2 and parts[1]:
            _candidate(state, parts[1])["product"] = True
        elif kind == "subpage" and len(parts) >= 3 and parts[1]:
            _candidate(state, parts[1])["subpages"].add(parts[2])
        elif kind == "option" and len(parts) >= 4 and parts[1]:
            _candidate(state, parts[1])["options"].add(parts[2])
        elif kind == "constraint" and len(parts) >= 4 and parts[1]:
            _candidate(state, parts[1])["hard_gates"][parts[2]] = parts[3]


def quality_evidence_potential(state: Mapping[str, object]) -> tuple[float, dict]:
    """Score the best hard-gate-qualified candidate, not exploration volume.

    The scorer consumes only public Environment v2.1 progress keys.  Opening
    additional products cannot increase the score once a better candidate is
    known, and option/subpage evidence counts only for a candidate that passes
    both category and budget gates.
    """
    discovery = 1.0 if state.get("result_set_seen") else 0.0
    candidates = state.get("candidates")
    if not isinstance(candidates, Mapping):
        candidates = {}
    best = {
        "product": 0.0,
        "category_pass": 0.0,
        "budget_pass": 0.0,
        "hard_gate_pair": 0.0,
        "option": 0.0,
        "subpage": 0.0,
    }
    best_value = 0.0
    for raw in candidates.values():
        if not isinstance(raw, Mapping):
            continue
        gates = raw.get("hard_gates")
        gates = gates if isinstance(gates, Mapping) else {}
        category_pass = float(gates.get("category") == "pass")
        budget_pass = float(gates.get("budget") == "pass")
        qualified = category_pass == 1.0 and budget_pass == 1.0
        options = raw.get("options")
        subpages = raw.get("subpages")
        components = {
            "product": float(bool(raw.get("product"))),
            "category_pass": category_pass,
            "budget_pass": budget_pass,
            "hard_gate_pair": float(qualified),
            "option": float(qualified and bool(options)),
            "subpage": (
                min(len(subpages), 2) / 2.0
                if qualified and isinstance(subpages, (set, frozenset))
                else 0.0
            ),
        }
        value = (
            0.05 * components["product"]
            + 0.20 * components["category_pass"]
            + 0.20 * components["budget_pass"]
            + 0.15 * components["hard_gate_pair"]
            + 0.20 * components["option"]
            + 0.20 * components["subpage"]
        )
        if value > best_value:
            best_value = value
            best = components
    potential = 0.05 * discovery + 0.95 * best_value
    return float(min(max(potential, 0.0), 1.0)), {
        "discovery": discovery,
        **best,
    }


def _validated_parameters(
    caps: Mapping[str, object] | None,
    weights: Mapping[str, object] | None,
) -> tuple[dict[str, int], dict[str, float]]:
    cap_values = {**DEFAULT_CAPS, **(dict(caps) if caps else {})}
    weight_values = {**DEFAULT_WEIGHTS, **(dict(weights) if weights else {})}
    clean_caps = {}
    clean_weights = {}
    for key in EVIDENCE_KEYS:
        clean_caps[key] = _non_negative_int(cap_values[key], f"caps.{key}")
        if clean_caps[key] < 1:
            raise ValueError(f"caps.{key} must be positive")
        try:
            clean_weights[key] = float(weight_values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weights.{key} must be numeric") from exc
        if not math.isfinite(clean_weights[key]) or clean_weights[key] < 0:
            raise ValueError(f"weights.{key} must be finite and non-negative")
    if sum(clean_weights.values()) <= 0:
        raise ValueError("at least one evidence weight must be positive")
    return clean_caps, clean_weights


def evidence_potential(
    snapshot: Mapping[str, object],
    *,
    caps: Mapping[str, object] | None = None,
    weights: Mapping[str, object] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return a bounded [0, 1] potential and its normalized components."""
    clean = sanitize_progress_snapshot(snapshot)
    clean_caps, clean_weights = _validated_parameters(caps, weights)
    components = {
        key: min(clean["evidence_counts"][key] / clean_caps[key], 1.0)
        for key in EVIDENCE_KEYS
    }
    denominator = sum(clean_weights.values())
    potential = sum(
        clean_weights[key] * components[key] for key in EVIDENCE_KEYS
    ) / denominator
    return float(potential), components


def build_evidence_shadow(
    steps: Sequence[object],
    *,
    caps: Mapping[str, object] | None = None,
    weights: Mapping[str, object] | None = None,
    max_turn_delta: float = 0.25,
) -> dict:
    """Build an auditable shadow trace without producing a training reward."""
    try:
        delta_cap = float(max_turn_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_turn_delta must be numeric") from exc
    if not math.isfinite(delta_cap) or delta_cap <= 0:
        raise ValueError("max_turn_delta must be finite and positive")

    zero = {
        "step_count": 0,
        "consecutive_repeats": 0,
        "no_progress_steps": 0,
        "effective_result_sets": 0,
        "opened_candidate_count": 0,
        "evidence_counts": {key: 0 for key in EVIDENCE_KEYS},
    }
    previous = zero
    quality_state = _empty_quality_state()
    previous_potential = 0.0
    trace = []
    errors = []

    for index, raw_step in enumerate(steps):
        step = raw_step if isinstance(raw_step, Mapping) else {}
        raw_progress = step.get("progress")
        if raw_progress is None:
            current = previous
            potential = previous_potential
            _, count_components = evidence_potential(
                current, caps=caps, weights=weights
            )
            _, components = quality_evidence_potential(quality_state)
            status = "missing"
        else:
            try:
                candidate = sanitize_progress_snapshot(raw_progress)
            except ValueError as exc:
                errors.append({"step": index, "reason": str(exc)})
                current = previous
                potential = previous_potential
                _, count_components = evidence_potential(
                    current, caps=caps, weights=weights
                )
                _, components = quality_evidence_potential(quality_state)
                status = "invalid"
            else:
                regressed = [
                    key
                    for key in EVIDENCE_KEYS
                    if candidate["evidence_counts"][key]
                    < previous["evidence_counts"][key]
                ]
                if regressed:
                    errors.append(
                        {
                            "step": index,
                            "reason": "evidence counter regression: "
                            + ",".join(regressed),
                        }
                    )
                    current = previous
                    potential = previous_potential
                    _, count_components = evidence_potential(
                        current, caps=caps, weights=weights
                    )
                    _, components = quality_evidence_potential(quality_state)
                    status = "regressed"
                else:
                    current = candidate
                    count_potential, count_components = evidence_potential(
                        current, caps=caps, weights=weights
                    )
                    _apply_quality_evidence(
                        quality_state, current["credited_evidence_added"]
                    )
                    potential, components = quality_evidence_potential(
                        quality_state
                    )
                    status = "ok"

        if raw_progress is None or status in {"invalid", "regressed"}:
            count_potential, count_components = evidence_potential(
                current, caps=caps, weights=weights
            )

        raw_delta = max(float(potential - previous_potential), 0.0)
        clipped_delta = min(raw_delta, delta_cap)
        trace.append(
            {
                "step": index,
                "status": status,
                "potential_before": previous_potential,
                "potential_after": potential,
                "raw_delta": raw_delta,
                "clipped_delta": clipped_delta,
                "components": components,
                "count_potential": count_potential,
                "count_components": count_components,
            }
        )
        previous = current
        previous_potential = potential

    positive = sum(row["clipped_delta"] > 0 for row in trace)
    return {
        "version": SHADOW_REWARD_VERSION,
        "shadow_only": True,
        "changes_sequence_reward": False,
        "valid": not errors,
        "errors": errors,
        "turns": len(trace),
        "positive_turns": positive,
        "zero_delta_turns": len(trace) - positive,
        "positive_turn_ratio": positive / len(trace) if trace else 0.0,
        "potential_final": previous_potential,
        "cumulative_clipped_delta": sum(
            row["clipped_delta"] for row in trace
        ),
        "trace": trace,
    }
