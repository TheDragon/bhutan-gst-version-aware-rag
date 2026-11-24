from __future__ import annotations

import re
from datetime import date


DATE_PATTERN = re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b")


def parse_query_date(query: str) -> str | None:
    match = DATE_PATTERN.search(query)
    if match:
        year, month, day = map(int, match.groups())
        return date(year, month, day).isoformat()
    year_match = re.search(r"\b(20\d{2})\b", query)
    return f"{year_match.group(1)}-12-31" if year_match else None


def valid_on(record: dict, target_date: str | None) -> bool:
    if not target_date:
        return True
    start = record.get("effective_from") or "0001-01-01"
    end = record.get("effective_to") or "9999-12-31"
    return start <= target_date <= end


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

