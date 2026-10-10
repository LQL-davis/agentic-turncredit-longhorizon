"""Verifier-preserving turn credit from audited public evidence deltas."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from shopping_grpo.training.grpo.evidence_reward import SHADOW_REWARD_VERSION
from shopping_grpo.training.grpo.trace import trace_turn_layout


def align_evidence_turn_deltas(
    response_mask: Sequence[int],
    response_attention: Sequence[int],
    turn_deltas: Sequence[float],
    evidence_turn_mask: Sequence[bool] | None = None,
) -> tuple[list[tuple[int, int]], list[float], int]:
    """Align environment-step deltas with all generated assistant turns.

    ``trace_turn_layout`` intentionally returns only assistant turns followed
    by visible tool output.  Agent-loop rollouts may additionally end with an
    assistant span: either a terminal tool call whose environment response is
    empty, or assistant-only final text that never executed an environment
    step.  The former consumes the final shadow delta; the latter receives a
    zero evidence delta.  Any other count mismatch still fails closed.
    """
    paired = [span for span, _ in trace_turn_layout(response_mask, response_attention)]
    try:
        active_end = response_attention.index(0)
    except ValueError:
        active_end = len(response_attention)
    terminal = None
    if active_end and response_mask[active_end - 1]:
        start = active_end - 1
        while start and response_mask[start - 1]:
            start -= 1
        terminal = (start, active_end)
    layout = [*paired, *([terminal] if terminal is not None else [])]
    deltas = [float(value) for value in turn_deltas]
    if evidence_turn_mask is not None:
        turn_mask = list(evidence_turn_mask)
        if any(type(value) is not bool for value in turn_mask):
            raise ValueError("evidence turn mask must contain booleans")
        unmatched = 0
        if terminal is not None and len(turn_mask) == len(layout) - 1:
            turn_mask.append(False)
            unmatched = 1
        if len(turn_mask) != len(layout):
            raise ValueError(
                "evidence turn mask does not match assistant/tool boundaries "
                f"(mask={len(turn_mask)}, paired={len(paired)}, "
                f"terminal={int(terminal is not None)})"
            )
        if sum(turn_mask) != len(deltas):
            raise ValueError(
                "evidence turn mask does not match environment deltas "
                f"(executed={sum(turn_mask)}, deltas={len(deltas)})"
            )
        iterator = iter(deltas)
        aligned = [next(iterator) if executed else 0.0 for executed in turn_mask]
        return layout, aligned, unmatched
    if len(deltas) == len(layout):
        return layout, deltas, 0
    if terminal is not None and len(deltas) == len(layout) - 1:
        return layout, [*deltas, 0.0], 1
    raise ValueError(
        "evidence turns do not match assistant/tool boundaries "
        f"(deltas={len(deltas)}, paired={len(paired)}, "
        f"terminal={int(terminal is not None)})"
    )


def evidence_turn_multipliers(
    deltas: Sequence[float],
    *,
    outcome_advantage: float,
    strength: float,
    min_multiplier: float,
    max_multiplier: float,
) -> list[float]:
    """Return positive, sign-preserving per-turn advantage multipliers.

    Positive group-relative trajectories emphasize evidence-producing turns.
    Negative trajectories protect those turns and put more negative weight on
    no-progress decisions.  The final decision is never weakened below the
    original sequence advantage.
    """
    values = [float(value) for value in deltas]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("evidence deltas must be finite and non-negative")
    parameters = (outcome_advantage, strength, min_multiplier, max_multiplier)
    if any(not math.isfinite(float(value)) for value in parameters):
        raise ValueError("evidence credit parameters must be finite")
    if strength < 0 or min_multiplier <= 0 or max_multiplier < min_multiplier:
        raise ValueError("invalid evidence credit multiplier bounds")
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = math.sqrt(variance)
    if deviation <= 1.0e-12 or outcome_advantage == 0:
        return [1.0] * len(values)
    direction = 1.0 if outcome_advantage > 0 else -1.0
    multipliers = []
    for value in values:
        z_score = max(min((value - mean) / deviation, 2.0), -2.0)
        multiplier = 1.0 + strength * direction * z_score
        multipliers.append(
            max(min(multiplier, max_multiplier), min_multiplier)
        )
    multipliers[-1] = max(multipliers[-1], 1.0)
    return multipliers


def _mass_preserving_multipliers(
    multipliers: Sequence[float],
    advantage_mass: Sequence[float],
    *,
    min_multiplier: float,
    max_multiplier: float,
) -> list[float]:
    """Project multipliers onto a bounded, weighted-mean-one surface.

    A common positive divisor preserves the relative turn ordering.  Clipping
    is applied inside the projection, and a monotone binary search chooses the
    divisor whose advantage-mass-weighted mean multiplier is one.  This keeps
    the rollout's original aggregate advantage while redistributing credit.
    """
    values = [float(value) for value in multipliers]
    weights = [float(value) for value in advantage_mass]
    if len(values) != len(weights):
        raise ValueError("multiplier and advantage-mass lengths must match")
    if not values:
        return []
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("evidence multipliers must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("turn advantage mass must be finite and non-negative")
    if not 0 < min_multiplier <= 1 <= max_multiplier:
        raise ValueError("mass preservation requires min <= 1 <= max")
    total_mass = sum(weights)
    if total_mass <= 1.0e-12:
        return [1.0] * len(values)

    def project(divisor: float) -> list[float]:
        return [
            max(min(value / divisor, max_multiplier), min_multiplier)
            for value in values
        ]

    lower = min(value / max_multiplier for value in values)
    upper = max(value / min_multiplier for value in values)
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        projected = project(midpoint)
        weighted_mean = sum(
            weight * value for weight, value in zip(weights, projected, strict=True)
        ) / total_mass
        if weighted_mean > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    return project((lower + upper) / 2.0)


def evidence_weighted_token_advantages(
    *,
    base_advantages: Sequence[float],
    response_mask: Sequence[int],
    response_attention: Sequence[int],
    turn_deltas: Sequence[float],
    strength: float,
    min_multiplier: float,
    max_multiplier: float,
    preserve_advantage_mass: bool = False,
) -> tuple[list[float], list[float]]:
    """Multiply assistant-token advantages while leaving tool tokens masked."""
    if not (
        len(base_advantages) == len(response_mask) == len(response_attention)
    ):
        raise ValueError("advantage, mask and attention lengths must match")
    layout, aligned_deltas, _ = align_evidence_turn_deltas(
        response_mask, response_attention, turn_deltas
    )
    active = [
        float(value)
        for value, mask, attention in zip(
            base_advantages, response_mask, response_attention, strict=True
        )
        if mask and attention
    ]
    outcome = active[0] if active else 0.0
    if any(value * outcome < 0 for value in active):
        raise ValueError("base advantages change sign within one rollout")
    multipliers = evidence_turn_multipliers(
        aligned_deltas,
        outcome_advantage=outcome,
        strength=strength,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    if preserve_advantage_mass:
        turn_mass = [
            sum(
                abs(float(base_advantages[index]))
                for index in range(start, end)
                if response_mask[index] and response_attention[index]
            )
            for start, end in layout
        ]
        multipliers = _mass_preserving_multipliers(
            multipliers,
            turn_mass,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
    result = [0.0] * len(base_advantages)
    for (start, end), multiplier in zip(
        layout, multipliers, strict=True
    ):
        for index in range(start, end):
            value = float(base_advantages[index]) * multiplier
            if value * float(base_advantages[index]) < 0:
                raise AssertionError("evidence credit changed advantage sign")
            result[index] = value
    return result, multipliers


def _validated_shadow(info: object, index: int) -> Mapping[str, object]:
    if not isinstance(info, Mapping):
        raise ValueError(f"shopping info at index {index} is not an object")
    shadow = info.get("training_reward_shadow")
    if not isinstance(shadow, Mapping):
        raise ValueError(f"shopping info at index {index} is missing evidence shadow")
    if shadow.get("version") != SHADOW_REWARD_VERSION:
        raise ValueError(f"shopping info at index {index} has the wrong shadow version")
    if (
        shadow.get("shadow_only") is not True
        or shadow.get("changes_sequence_reward") is not False
        or shadow.get("valid") is not True
    ):
        raise ValueError(f"shopping info at index {index} has an unsafe evidence shadow")
    trace = shadow.get("trace")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise ValueError(f"shopping info at index {index} has an invalid shadow trace")
    return shadow


def apply_evidence_credit_advantages(
    batch,
    *,
    strength: float,
    min_multiplier: float,
    max_multiplier: float,
    preserve_advantage_mass: bool = False,
) -> dict[str, float]:
    """Apply audited evidence multipliers to a veRL actor-update batch."""
    import torch

    shopping = batch.non_tensor_batch.get("shopping")
    if shopping is None or len(shopping) != len(batch.batch["responses"]):
        raise ValueError("evidence credit requires one shopping record per rollout")
    response_length = batch.batch["responses"].shape[1]
    response_attention = batch.batch["attention_mask"][:, -response_length:]
    mixed_rows = []
    all_deltas = []
    all_multipliers = []
    unmatched_terminal_turns = 0
    non_evidence_turns = 0
    advantage_mass_ratios = []
    for index, info in enumerate(shopping):
        shadow = _validated_shadow(info, index)
        deltas = [float(row["clipped_delta"]) for row in shadow["trace"]]
        turn_mask = info.get("evidence_turn_mask")
        if not isinstance(turn_mask, Sequence) or isinstance(
            turn_mask, (str, bytes)
        ):
            raise ValueError(
                f"shopping info at index {index} is missing evidence turn mask"
            )
        _, aligned_deltas, unmatched = align_evidence_turn_deltas(
            batch.batch["response_mask"][index].tolist(),
            response_attention[index].tolist(),
            deltas,
            turn_mask,
        )
        mixed, multipliers = evidence_weighted_token_advantages(
            base_advantages=batch.batch["advantages"][index].tolist(),
            response_mask=batch.batch["response_mask"][index].tolist(),
            response_attention=response_attention[index].tolist(),
            turn_deltas=aligned_deltas,
            strength=strength,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            preserve_advantage_mass=preserve_advantage_mass,
        )
        base_mass = sum(
            abs(float(value))
            for value, mask, attention in zip(
                batch.batch["advantages"][index].tolist(),
                batch.batch["response_mask"][index].tolist(),
                response_attention[index].tolist(),
                strict=True,
            )
            if mask and attention
        )
        mixed_mass = sum(abs(float(value)) for value in mixed)
        advantage_mass_ratios.append(
            mixed_mass / base_mass if base_mass > 1.0e-12 else 1.0
        )
        mixed_rows.append(mixed)
        all_deltas.extend(aligned_deltas)
        all_multipliers.extend(multipliers)
        unmatched_terminal_turns += unmatched
        non_evidence_turns += sum(not value for value in turn_mask) + unmatched
    mixed = torch.tensor(
        mixed_rows,
        dtype=batch.batch["advantages"].dtype,
        device=batch.batch["advantages"].device,
    )
    batch.batch["advantages"] = mixed
    batch.batch["returns"] = mixed
    return {
        "evidence_credit/turns": float(len(all_deltas)),
        "evidence_credit/positive_delta_ratio": (
            float(sum(value > 0 for value in all_deltas) / len(all_deltas))
            if all_deltas else 0.0
        ),
        "evidence_credit/multiplier_mean": (
            float(sum(all_multipliers) / len(all_multipliers))
            if all_multipliers else 1.0
        ),
        "evidence_credit/multiplier_min": (
            float(min(all_multipliers)) if all_multipliers else 1.0
        ),
        "evidence_credit/multiplier_max": (
            float(max(all_multipliers)) if all_multipliers else 1.0
        ),
        "evidence_credit/unmatched_terminal_turns": float(
            unmatched_terminal_turns
        ),
        "evidence_credit/non_evidence_turns": float(non_evidence_turns),
        "evidence_credit/preserve_advantage_mass": float(preserve_advantage_mass),
        "evidence_credit/advantage_mass_ratio_mean": (
            float(sum(advantage_mass_ratios) / len(advantage_mass_ratios))
            if advantage_mass_ratios else 1.0
        ),
        "evidence_credit/advantage_mass_ratio_max_abs_error": (
            float(max(abs(value - 1.0) for value in advantage_mass_ratios))
            if advantage_mass_ratios else 0.0
        ),
    }
