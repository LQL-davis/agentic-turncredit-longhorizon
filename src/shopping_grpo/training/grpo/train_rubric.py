"""Query-only, frozen training Rubrics; never read task facts or a gold product."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from shopping_grpo.evaluation.rubric import extract_price_candidates


TRAIN_RUBRIC_VERSION = "shopping-train-rubric-v1"
_CATEGORY = re.compile(
    r"(?:想|要|请|帮我|求)?(?:买|找|购|选|推荐|入手|预订)"
    r"(?:一款|一个|一台|一套|一件|一块|一双|一支|一盒|一组)?"
    r"(?:[^，。；]{0,32}?的)?"
    r"([\u4e00-\u9fff]{1,9}(?:喷头|打印机|鱼肝油|水彩笔|咖啡|器|镜|灯|鞋|裤|包|琴|油|笔|杆|线|板|卡|粮|座椅))"
    r"(?=[，。；、\s]|$)"
)
_BRAND = re.compile(
    r"(?:品牌(?:是|为|要|选|指定)|(?:要|选|买|找)([A-Za-z\u4e00-\u9fff]{2,6})品牌)"
    r"([A-Za-z\u4e00-\u9fff]{2,8})?"
)
_MODEL = re.compile(r"型号(?:是|为|要|选)?\s*([A-Za-z0-9][A-Za-z0-9._-]{2,20})", re.I)
_FUNCTION = re.compile(r"(?:带|有|具有)([^，。；、带有]{2,12}?)功能")
_OPTION = re.compile(
    r"(颜色|尺码|色号)(?:要|为|选|是)\s*([A-Za-z0-9\u4e00-\u9fff]{1,8})"
    r"(?=[，。；、\s的]|$)", re.I
)
_EXACT_PRICE = re.compile(
    r"(?:预算|价格|售价|价位|定价)(?:为|是|在|约|控制在)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:元|块)(?:左右|上下)?"
)
_PRICE_CONTEXT = re.compile(r"预算|价格|售价|价位|定价|预期花费")
_CHINESE_PRICE = re.compile(r"([零一二两三四五六七八九十百千万]+)\s*(元|块)(?:钱)?")
_CHINESE_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                   "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _chinese_amount(text: str) -> int:
    total = 0
    section = 0
    digit = 0
    for character in text:
        if character in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[character]
        elif character == "万":
            total += (section + digit) * 10000
            section = digit = 0
        else:
            section += (digit or 1) * _CHINESE_UNITS[character]
            digit = 0
    return total + section + digit


def user_query(raw_prompt: object) -> str:
    """Extract only the user message the Actor receives, never private metadata."""
    if not isinstance(raw_prompt, Sequence) or isinstance(raw_prompt, (str, bytes)):
        raise ValueError("raw_prompt must be a sequence of messages")
    users = [
        message.get("content")
        for message in raw_prompt
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if len(users) != 1 or not isinstance(users[0], str):
        raise ValueError("training Rubric requires exactly one textual user Query")
    query = re.sub(r"^\s*Instruction\s*:\s*", "", users[0], flags=re.I).strip()
    if not query or len(query) > 4096:
        raise ValueError("training Query must be non-empty and bounded")
    return query


def _candidate(
    query: str, *, dimension: str, expected: object, field_path: str,
    operator: str, start: int, end: int, hardness: str = "hard",
) -> dict:
    if not 0 <= start < end <= len(query):
        raise ValueError("candidate span is outside the Query")
    return {
        "constraint_type": dimension,
        "field_path": field_path,
        "operator": operator,
        "expected_value": expected,
        "hardness_hint": hardness,
        "query_spans": [{"text": query[start:end], "start": start, "end": end}],
        "data_sources": ["query"],
    }


def build_train_rubric(task_id: int, raw_prompt: object) -> dict:
    """Freeze conservative, auditable candidates from a training prompt.

    Ambiguous requirements are deliberately omitted. Missing dimensions are
    measured as coverage gaps rather than filled with hidden target fields.
    """
    query = user_query(raw_prompt)
    constraints: list[dict] = []
    price_spans: list[tuple[int, int]] = []
    for price in extract_price_candidates(query):
        span = price["query_spans"][0]
        left = max(0, span["start"] - 10)
        before = query[left:span["start"]]
        currency_suffix = query[span["end"]:span["end"] + 3]
        # The evaluation extractor deliberately returns a superset. A training
        # Rubric must not turn dimensions such as 10-46 gauge or 5斤 into CNY.
        if not ("元" in span["text"] or "块" in span["text"]
                or currency_suffix.startswith(("元", "块"))
                or _PRICE_CONTEXT.search(before)):
            continue
        price_spans.append((span["start"], span["end"]))
        constraints.append({
            key: price[key]
            for key in (
                "constraint_type", "field_path", "operator", "expected_value",
                "hardness_hint", "query_spans", "data_sources",
            )
        })
    for match in _EXACT_PRICE.finditer(query):
        if any(match.start() < right and left < match.end() for left, right in price_spans):
            continue
        constraints.append(_candidate(
            query, dimension="price_preference",
            expected={"value": float(match.group(1)), "currency": "CNY"},
            field_path="purchase.price", operator="approximately",
            start=match.start(), end=match.end(), hardness="soft",
        ))
    for match in _CHINESE_PRICE.finditer(query):
        amount = _chinese_amount(match.group(1))
        if amount <= 0 or any(match.start() < right and left < match.end() for left, right in price_spans):
            continue
        left = query[max(0, match.start() - 12):match.start()]
        right = query[match.end():match.end() + 4]
        if re.search(r"不超过|不能超过|不高于|最高|封顶", left) or re.match(r"(?:以?内|以下|之内)", right):
            dimension, operator, hardness = "budget_upper", "lte", "hard"
        elif re.search(r"不低于|至少|最低", left) or re.match(r"以上", right):
            dimension, operator, hardness = "price_lower", "gte", "hard"
        else:
            dimension, operator, hardness = "price_preference", "approximately", "soft"
        constraints.append(_candidate(
            query, dimension=dimension, expected={"value": float(amount), "currency": "CNY"},
            field_path="purchase.price", operator=operator,
            start=match.start(), end=match.end(), hardness=hardness,
        ))

    category = _CATEGORY.search(query)
    if category:
        value = category.group(1)
        # The noun after the last 的 is the requested product; preceding text
        # is often usage or material, not part of the category name.
        if "的" in value:
            value = value.rsplit("的", 1)[-1]
        start = category.end(1) - len(value)
        constraints.append(_candidate(
            query, dimension="category", expected=value,
            field_path="product.category", operator="in_category",
            start=start, end=category.end(1), hardness="soft",
        ))
    for match in _BRAND.finditer(query):
        value = match.group(1) or match.group(2)
        if value and value not in {"质量", "保证", "正品", "官方"}:
            group = 1 if match.group(1) else 2
            constraints.append(_candidate(
                query, dimension="brand", expected=value,
                field_path="product.brand", operator="eq",
                start=match.start(group), end=match.end(group),
            ))
    for match in _MODEL.finditer(query):
        constraints.append(_candidate(
            query, dimension="model", expected=match.group(1),
            field_path="product.model", operator="eq",
            start=match.start(1), end=match.end(1),
        ))
    for match in _FUNCTION.finditer(query):
        value = match.group(1).strip()
        if value and not any(word in value for word in ("什么", "哪些", "多种", "多个")):
            constraints.append(_candidate(
                query, dimension="core_function", expected=value,
                field_path="product.attributes", operator="contains",
                start=match.start(1), end=match.end(1),
            ))
    for match in _OPTION.finditer(query):
        raw = match.group(2)
        value = raw.rstrip("的")
        if "的" in value:
            value = value.rsplit("的", 1)[-1]
        if not value:
            continue
        start = match.end(2) - len(value) - (len(raw) - len(raw.rstrip("的")))
        constraints.append(_candidate(
            query, dimension="option", expected=value,
            field_path=f"purchase.options.{match.group(1)}", operator="eq",
            start=start, end=start + len(value),
        ))

    unique: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in sorted(constraints, key=lambda row: row["query_spans"][0]["start"]):
        identity = (item["constraint_type"], item["field_path"], str(item["expected_value"]))
        if identity not in seen:
            seen.add(identity)
            unique.append({"rubric_id": f"r{len(unique)+1:04d}", **item})

    rubric = {
        "schema_version": TRAIN_RUBRIC_VERSION,
        "task_id": int(task_id),
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "source": "actor_user_query_only",
        "constraints": unique,
    }
    validate_train_rubric(rubric, query)
    return rubric


def validate_train_rubric(rubric: Mapping, query: str) -> None:
    """Reject invented values, malformed spans and private data-source labels."""
    if rubric.get("schema_version") != TRAIN_RUBRIC_VERSION:
        raise ValueError("unsupported training Rubric version")
    if rubric.get("source") != "actor_user_query_only":
        raise ValueError("training Rubric source is not Query-only")
    if rubric.get("query_hash") != hashlib.sha256(query.encode("utf-8")).hexdigest():
        raise ValueError("training Rubric Query hash mismatch")
    if not isinstance(rubric.get("constraints"), list):
        raise ValueError("training Rubric constraints must be a list")
    ids: set[str] = set()
    for item in rubric["constraints"]:
        if not isinstance(item, Mapping) or item.get("data_sources") != ["query"]:
            raise ValueError("training Rubric may only cite the user Query")
        if item.get("rubric_id") in ids:
            raise ValueError("duplicate training rubric_id")
        ids.add(item["rubric_id"])
        spans = item.get("query_spans")
        if not isinstance(spans, list) or len(spans) != 1:
            raise ValueError("training Rubric must have one Query span")
        span = spans[0]
        if query[span["start"]:span["end"]] != span["text"]:
            raise ValueError("training Rubric span does not match the Query")
        value = item.get("expected_value")
        if isinstance(value, str) and value not in span["text"]:
            raise ValueError("training Rubric value is absent from Query span")
