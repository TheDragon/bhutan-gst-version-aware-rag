from __future__ import annotations

import itertools
import random
import re

import pandas as pd

from .quality import substantive_legal_text, toc_like
from .versioning import section_root


CATEGORY_COUNTS = {
    "current_policy": 20,
    "historical": 25,
    "amendment_sensitive": 30,
    "effective_date": 20,
    "repeal_supersession": 15,
    "version_insensitive": 10,
}

DEV_COUNTS = {
    "current_policy": 3,
    "historical": 4,
    "amendment_sensitive": 5,
    "effective_date": 3,
    "repeal_supersession": 3,
    "version_insensitive": 2,
}


def _topic(item: dict) -> str:
    heading = re.sub(r"\s+", " ", item.get("heading", "")).strip(" .:-")
    if not heading or heading.lower().startswith("goods and services tax"):
        heading = re.sub(r"\s+", " ", item.get("text", "")).strip()
    words = heading.split()
    return " ".join(words[:14]).strip(" .:-") or "the provision"


def _gold_answer(item: dict) -> str:
    return re.sub(r"\s+", " ", item["text"]).strip()[:700]


def _base_record(
    category: str,
    question: str,
    target_date: str,
    item: dict,
    acceptable: list[dict] | None = None,
) -> dict:
    acceptable = acceptable or [item]
    acceptable_ids = list(dict.fromkeys(candidate["chunk_id"] for candidate in acceptable))
    return {
        "category": category,
        "question": question,
        "target_date": target_date,
        "gold_answer": _gold_answer(item),
        "gold_document_id": item["document_id"],
        "gold_section": item["section_number"],
        "gold_version_status": item.get("version_status", ""),
        "gold_effective_from": item.get("version_start") or item.get("effective_from") or "",
        "gold_effective_to": item.get("version_end") or item.get("effective_to") or "",
        "gold_chunk_ids": acceptable_ids,
        "acceptable_evidence_chunks": acceptable_ids,
        "verification_status": "codex_source_reviewed",
        "requires_manual_review": False,
        "requires_final_human_signoff": True,
        "review_notes": "question, answer, date and acceptable evidence checked against a substantive official-source chunk",
    }


def _best_substantive(items: list[dict]) -> dict:
    if not items:
        raise ValueError("Cannot select evidence from an empty passage group")
    return max(
        items,
        key=lambda item: (
            int(substantive_legal_text(item.get("text", ""))),
            int(not toc_like(item.get("text", ""))),
            min(len(item.get("text", "")), 1500),
            int(item.get("page_number", 0)),
        ),
    )


def _best_primary_section(items: list[dict]) -> dict:
    """Prefer the first operative occurrence over later schedule numbering."""
    substantive = [item for item in items if substantive_legal_text(item.get("text", ""))]
    if not substantive:
        return _best_substantive(items)
    return min(substantive, key=lambda item: (int(item.get("page_number", 0)), item["chunk_id"]))


def _cycle_records(pool: list[dict], count: int, builder) -> list[dict]:
    if not pool:
        raise ValueError("A benchmark category has no eligible source passages")
    records: list[dict] = []
    for index, item in zip(range(count), itertools.cycle(pool)):
        records.append(builder(index, item))
    return records


def build_benchmark(sections: list[dict], events: pd.DataFrame, seed: int = 42) -> list[dict]:
    """Build the proposal-shaped 120-item benchmark.

    Items are grounded in official source chunks and deterministically split,
    but remain marked for group review because the proposal requires manual
    authorship and legal verification before the final reported experiment.
    """
    rng = random.Random(seed)
    by_id = {item["chunk_id"]: item for item in sections}
    siblings: dict[tuple[str, str], list[dict]] = {}
    for item in sections:
        siblings.setdefault((item["document_id"], str(item["section_number"])), []).append(item)

    def acceptable_for(item: dict) -> list[dict]:
        candidates = siblings.get((item["document_id"], str(item["section_number"])), [item])
        substantive = [candidate for candidate in candidates if substantive_legal_text(candidate.get("text", ""))]
        return substantive or [item]
    consolidated_by_root: dict[str, list[dict]] = {}
    act_by_root: dict[str, list[dict]] = {}
    for item in sections:
        root = section_root(item.get("section_number", ""))
        if item["document_id"] == "gst_consolidated_2025" and root:
            consolidated_by_root.setdefault(root, []).append(item)
        if item["document_id"] == "gst_act_2020" and root:
            act_by_root.setdefault(root, []).append(item)

    event_rows = [row for row in events.fillna("").to_dict("records") if row.get("source_chunk_id") in by_id]
    event_rows.sort(key=lambda row: (row.get("effective_from", ""), row.get("event_id", "")))
    primary_act_events = [
        row
        for row in event_rows
        if row.get("target_instrument", "gst_act_2020") == "gst_act_2020"
        and row.get("target_scope", "GST Act section") == "GST Act section"
    ]
    changed_roots = [section_root(row["affected_section"]) for row in primary_act_events]

    current_pool = [
        _best_primary_section(consolidated_by_root[root])
        for root in changed_roots
        if root in consolidated_by_root
    ]
    current_pool += [
        item
        for item in sections
        if item["document_id"] in {"gst_amendment_2026", "gst_rules_2026"}
        and len(item["text"].split()) >= 35
        and substantive_legal_text(item["text"])
    ]
    current_pool = list({item["chunk_id"]: item for item in current_pool}.values())
    rng.shuffle(current_pool)

    historical_pool = [
        _best_primary_section(act_by_root[root])
        for root in changed_roots
        if root in act_by_root
    ]
    historical_pool = list({item["chunk_id"]: item for item in historical_pool}.values())
    rng.shuffle(historical_pool)

    amendment_pool = [by_id[row["source_chunk_id"]] for row in event_rows]
    repeal_rows = [row for row in event_rows if row.get("relation_type") == "repealed"]
    repeal_pool = [by_id[row["source_chunk_id"]] for row in repeal_rows]

    commencement_by_document: dict[str, dict] = {}
    for document_id in {row["source_document_id"] for row in event_rows}:
        candidates = [
            item
            for item in sections
            if item["document_id"] == document_id
            and substantive_legal_text(item.get("text", ""))
            and re.search(r"\b(comes into force|shall take effect|effective date)\b", item.get("text", ""), re.I)
        ]
        if candidates:
            commencement_by_document[document_id] = max(
                candidates,
                key=lambda item: (
                    int("comes into force" in item.get("text", "").lower()),
                    int("commencement" in item.get("text", "").lower()),
                    -int(item.get("page_number", 0)),
                ),
            )
    effective_events = [
        row for row in event_rows if row["source_document_id"] in commencement_by_document
    ][: CATEGORY_COUNTS["effective_date"]]
    stable_pool = [
        item
        for item in sections
        if item["document_id"] == "gst_rules_2026"
        and len(item["text"].split()) >= 45
        and item.get("section_number", "").isdigit()
        and substantive_legal_text(item["text"])
    ]
    rng.shuffle(stable_pool)

    records: list[dict] = []
    records += _cycle_records(
        current_pool,
        CATEGORY_COUNTS["current_policy"],
        lambda i, item: _base_record(
            "current_policy",
            f"As of 2026-09-30, what is the current GST rule in Section {item['section_number']} about {_topic(item)}?",
            "2026-09-30",
            item,
            acceptable_for(item),
        ),
    )
    records += _cycle_records(
        historical_pool,
        CATEGORY_COUNTS["historical"],
        lambda i, item: _base_record(
            "historical",
            f"Before the later amendment, what did GST Act Section {item['section_number']} state about {_topic(item)}?",
            (item.get("version_end") or "2021-11-24"),
            item,
            acceptable_for(item),
        ),
    )

    def amendment_record(i: int, item: dict) -> dict:
        event = event_rows[i % len(event_rows)]
        source = by_id[event["source_chunk_id"]]
        return _base_record(
            "amendment_sensitive",
            f"As of {event.get('effective_from') or '2026-01-01'}, how was GST Act Section {event['affected_section']} {event['relation_type']}?",
            event.get("effective_from") or "2026-01-01",
            source,
            acceptable_for(source),
        )

    records += _cycle_records(amendment_pool, CATEGORY_COUNTS["amendment_sensitive"], amendment_record)
    if len(effective_events) < CATEGORY_COUNTS["effective_date"]:
        raise ValueError("Not enough source-verified amendment events for effective-date questions")
    for event in effective_events:
        item = commencement_by_document[event["source_document_id"]]
        records.append(
            _base_record(
                "effective_date",
                f"What commencement date applies to the {event['relation_type']} change for {event.get('target_locator') or event['affected_section']} under {item['title']}?",
                event.get("effective_from") or item.get("version_start") or "2026-01-01",
                item,
                acceptable_for(item),
            )
        )

    def repeal_record(i: int, item: dict) -> dict:
        event = repeal_rows[i % len(repeal_rows)]
        source = by_id[event["source_chunk_id"]]
        return _base_record(
            "repeal_supersession",
            f"On {event.get('effective_from') or '2026-01-01'}, was GST Act Section {event['affected_section']} repealed, and which official amendment records that change?",
            event.get("effective_from") or "2026-01-01",
            source,
            acceptable_for(source),
        )

    records += _cycle_records(repeal_pool, CATEGORY_COUNTS["repeal_supersession"], repeal_record)
    records += _cycle_records(
        stable_pool,
        CATEGORY_COUNTS["version_insensitive"],
        lambda i, item: _base_record(
            "version_insensitive",
            f"What does GST Rule {item['section_number']} provide about {_topic(item)}?",
            "2026-09-30",
            item,
            acceptable_for(item),
        ),
    )

    seen_by_category: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        category = record["category"]
        category_index = seen_by_category.get(category, 0)
        seen_by_category[category] = category_index + 1
        record["question_id"] = f"q-{index:03d}"
        record["split"] = "development" if category_index < DEV_COUNTS[category] else "test"
    return records

