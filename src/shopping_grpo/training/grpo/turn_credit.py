"""Public-page Rubric evidence and signed, centered Assistant-turn advantages."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence

from shopping_grpo.training.grpo.evidence_credit import align_evidence_turn_deltas
from shopping_grpo.training.grpo.train_rubric import TRAIN_RUBRIC_VERSION


TURN_CREDIT_VERSION = "shopping-turn-credit-v1"
_PRICE = re.compile(r"\d+(?:\.\d+)?")
_PAGE_FIELDS = {"page_type", "asin", "title", "brand", "category", "price", "key_attributes", "selected_options", "content"}


def parse_visible_page(observation: object) -> dict:
    """Keep only fields present in the Actor-visible observation text."""
    text = str(observation or "")
    if not text.startswith("[SHOPPING_OBSERVATION_V2]"):
        return {"page_type": "unknown"}
    result: dict[str, object] = {"page_type": "unknown"}
    products: list[str] = []
    for line in text.splitlines()[1:]:
        if line.count("|") == 6 and line.split("|", 1)[0].isdigit():
            products.append(line)
            continue
        key, sep, value = line.partition(": ")
        if sep and key in _PAGE_FIELDS:
            if key == "selected_options":
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = {}
                result[key] = parsed if isinstance(parsed, dict) else {}
            else:
                result[key] = value[:4096]
    if products:
        result["products"] = products[:20]
    return result


def _normalize(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _price_interval(value: object) -> tuple[float, float] | None:
    numbers = [float(number) for number in _PRICE.findall(str(value or ""))]
    if not numbers or len(numbers) > 2 or not all(math.isfinite(n) for n in numbers):
        return None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers), max(numbers)


def constraint_status(constraint: Mapping, page: Mapping) -> str:
    """Unknown unless a public field can actually support or refute the claim."""
    page_type = page.get("page_type")
    if page_type not in {"product_detail", "information_subpage"}:
        return "unknown"
    kind = constraint.get("constraint_type")
    expected = constraint.get("expected_value")
    if kind in {"budget_upper", "price_lower", "price_range", "price_preference"}:
        interval = _price_interval(page.get("price"))
        if interval is None or not isinstance(expected, Mapping):
            return "unknown"
        low, high = interval
        operator = constraint.get("operator")
        if operator == "lte":
            limit = float(expected["value"])
            return "supported" if high <= limit else "contradicted" if low > limit else "unknown"
        if operator == "gte":
            limit = float(expected["value"])
            return "supported" if low >= limit else "contradicted" if high < limit else "unknown"
        if operator == "between":
            bottom, top = float(expected["min"]), float(expected["max"])
            return "supported" if bottom <= low <= high <= top else "contradicted" if high < bottom or low > top else "unknown"
        if operator == "approximately":
            target = float(expected["value"])
            tolerance = max(target * 0.15, 3.0)
            return "supported" if target - tolerance <= low <= high <= target + tolerance else "contradicted" if high < target - tolerance or low > target + tolerance else "unknown"
        return "unknown"

    literal = _normalize(expected)
    if not literal:
        return "unknown"
    if kind == "brand":
        visible = _normalize(page.get("brand"))
        return "supported" if literal == visible else "contradicted" if visible else "unknown"
    if kind == "category":
        visible = _normalize(page.get("category")) + _normalize(page.get("title"))
        return "supported" if literal in visible else "unknown"
    if kind == "model":
        visible = _normalize(page.get("title")) + _normalize(page.get("key_attributes")) + _normalize(page.get("content"))
        return "supported" if literal in visible else "unknown"
    if kind == "core_function":
        visible = _normalize(page.get("key_attributes")) + _normalize(page.get("content"))
        return "supported" if literal in visible else "unknown"
    if kind == "option":
        axis = str(constraint.get("field_path", "")).rsplit(".", 1)[-1]
        selected = page.get("selected_options")
        if not isinstance(selected, Mapping):
            return "unknown"
        matching_axes = [key for key in selected if key == axis or axis in str(key)]
        if len(matching_axes) != 1:
            return "unknown"
        choice = selected[matching_axes[0]]
        if choice is None:
            return "unknown"
        return "supported" if literal in _normalize(choice) else "contradicted"
    return "unknown"


def score_turn_events(rubric: Mapping, events: Sequence[Mapping]) -> dict:
    """Score observed changes only; the final environment Reward is untouched."""
    if rubric.get("schema_version") != TRAIN_RUBRIC_VERSION:
        raise ValueError("invalid training Rubric for turn scoring")
    constraints = rubric.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError("training Rubric constraints are missing")
    deltas: list[float] = []
    evidence: list[dict] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError("turn event is not an object")
        status = event.get("status")
        if status not in {"executed", "guard_rejected", "terminal", "no_environment"}:
            raise ValueError(f"turn event {index} has incomplete status")
        before = event.get("before")
        after = event.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"turn event {index} lacks visible page evidence")
        tool = str(event.get("tool") or "")
        delta = -0.12 if status == "guard_rejected" else 0.0
        if event.get("repeat") is True:
            delta -= 0.06
        matched: list[str] = []
        contradicted: list[str] = []
        if status in {"executed", "terminal"}:
            for constraint in constraints:
                old = constraint_status(constraint, before)
                new = constraint_status(constraint, after)
                amount = 0.04 if constraint.get("hardness_hint") == "soft" else 0.08
                if new == "supported" and old != "supported":
                    delta += amount
                    matched.append(str(constraint["rubric_id"]))
                elif new == "contradicted" and old != "contradicted":
                    delta -= amount
                    contradicted.append(str(constraint["rubric_id"]))
        if tool == "search_products" and status == "executed" and after.get("page_type") == "search_results":
            searchable = " ".join(str(row) for row in after.get("products", []))
            if any(
                isinstance(c.get("expected_value"), str)
                and _normalize(c["expected_value"]) in _normalize(searchable)
                for c in constraints if c.get("constraint_type") in {"category", "brand", "model"}
            ):
                delta += 0.03
        if tool == "buy_now" and status == "terminal":
            gates = [
                constraint_status(c, before)
                for c in constraints
                if c.get("constraint_type") in {"category", "budget_upper", "price_range"}
            ]
            if "contradicted" in gates:
                delta -= 0.20
            elif gates and all(value == "supported" for value in gates):
                delta += 0.06
            elif gates:
                delta -= 0.06
        clipped = max(min(delta, 0.25), -0.25)
        deltas.append(clipped)
        evidence.append({"turn": index, "tool": tool, "delta": clipped, "supported": matched, "contradicted": contradicted})
    return {
        "version": TURN_CREDIT_VERSION,
        "query_hash": rubric["query_hash"],
        "rubric_version": TRAIN_RUBRIC_VERSION,
        "constraint_count": len(constraints),
        "deltas": deltas,
        "evidence": evidence,
    }


def centered_bounded_turn_deltas(scores: Sequence[float], token_counts: Sequence[int], cap: float) -> list[float]:
    if len(scores) != len(token_counts) or not scores:
        raise ValueError("turn scores and positive token counts must align")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("turn credit cap must be positive and finite")
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("turn score must be finite")
    if any(type(count) is not int or count <= 0 for count in token_counts):
        raise ValueError("turn token counts must be positive integers")
    clipped = [max(min(float(value), cap), -cap) for value in scores]
    weighted_mean = sum(value * count for value, count in zip(clipped, token_counts, strict=True)) / sum(token_counts)
    centered = [value - weighted_mean for value in clipped]
    maximum = max(abs(value) for value in centered)
    scale = min(1.0, cap / maximum) if maximum else 1.0
    return [value * scale for value in centered]


def mixed_token_advantages(
    *, base_advantages: Sequence[float], response_mask: Sequence[int],
    response_attention: Sequence[int], turn_deltas: Sequence[float],
    strength: float, max_turn_delta: float,
) -> tuple[list[float], list[float]]:
    """Return token advantages with zero-sum signed turn adjustments."""
    if not (len(base_advantages) == len(response_mask) == len(response_attention)):
        raise ValueError("advantage and masks must have equal length")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("turn credit strength must be finite and non-negative")
    if strength == 0:
        return [float(value) for value in base_advantages], []
    spans, scores, _ = align_evidence_turn_deltas(response_mask, response_attention, turn_deltas)
    if not spans:
        raise ValueError("rollout has no aligned Assistant turns")
    counts = [end - start for start, end in spans]
    centered = centered_bounded_turn_deltas(scores, counts, max_turn_delta)
    result = [0.0] * len(base_advantages)
    for (start, end), delta in zip(spans, centered, strict=True):
        for token in range(start, end):
            if not response_mask[token] or not response_attention[token]:
                raise ValueError("Assistant span covers a tool or padding token")
            result[token] = float(base_advantages[token]) + strength * delta
    return result, centered


def apply_turn_credit_advantages(batch, *, strength: float, max_turn_delta: float) -> dict[str, float]:
    """Modify only accepted GRPO Actor batches; misaligned rows fall back to base."""
    import torch

    shopping = batch.non_tensor_batch.get("shopping")
    if shopping is None or len(shopping) != len(batch.batch["responses"]):
        raise ValueError("turn credit requires one shopping record per rollout")
    response_length = batch.batch["responses"].shape[1]
    attention = batch.batch["attention_mask"][:, -response_length:]
    rows: list[list[float]] = []
    applied = 0
    fallbacks = 0
    signed: list[float] = []
    for row, info in enumerate(shopping):
        base = batch.batch["advantages"][row].tolist()
        payload = info.get("turn_credit") if isinstance(info, Mapping) else None
        if not isinstance(payload, Mapping) or payload.get("version") != TURN_CREDIT_VERSION or not payload.get("constraint_count"):
            rows.append(base)
            fallbacks += 1
            continue
        try:
            mixed, centered = mixed_token_advantages(
                base_advantages=base,
                response_mask=batch.batch["response_mask"][row].tolist(),
                response_attention=attention[row].tolist(),
                turn_deltas=payload["deltas"],
                strength=strength,
                max_turn_delta=max_turn_delta,
            )
        except (KeyError, TypeError, ValueError):
            rows.append(base)
            fallbacks += 1
            continue
        rows.append(mixed)
        signed.extend(centered)
        applied += 1
    mixed_tensor = torch.tensor(rows, dtype=batch.batch["advantages"].dtype, device=batch.batch["advantages"].device)
    batch.batch["advantages"] = mixed_tensor
    batch.batch["returns"] = mixed_tensor
    return {
        "turn_credit/applied_rollouts": float(applied),
        "turn_credit/fallback_rollouts": float(fallbacks),
        "turn_credit/positive_turn_ratio": float(sum(x > 0 for x in signed) / len(signed)) if signed else 0.0,
        "turn_credit/negative_turn_ratio": float(sum(x < 0 for x in signed) / len(signed)) if signed else 0.0,
        "turn_credit/turn_count": float(len(signed)),
    }
