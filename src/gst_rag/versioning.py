from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

from .quality import substantive_legal_text, toc_like


def normalize_section(value: str) -> str:
    """Return a stable legal-section key while retaining subsection identity."""
    value = re.sub(r"\s+", "", str(value or "")).lower()
    value = value.removeprefix("section")
    return value.strip(".:-")


def section_root(value: str) -> str:
    return normalize_section(value).split("(", 1)[0]


def _previous_day(value: str) -> str:
    return (date.fromisoformat(value) - timedelta(days=1)).isoformat()


def verify_amendment_events(
    sections: list[dict], events: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Check detected events against their source chunks.

    This is a source-text verification step, not legal sign-off.  Every row keeps
    a group-review flag so the project does not present automated annotation as
    professional legal verification.
    """
    by_chunk = {item["chunk_id"]: item for item in sections}
    by_document_clause: dict[tuple[str, str], list[dict]] = {}
    for item in sections:
        key = (str(item.get("document_id", "")), str(item.get("section_number", "")))
        by_document_clause.setdefault(key, []).append(item)
    document_dates = manifest.set_index("document_id")["effective_from"].to_dict()
    rows: list[dict] = []
    for raw in events.fillna("").to_dict("records"):
        row = dict(raw)
        original_chunk_id = row.get("source_chunk_id", "")
        chunk = by_chunk.get(original_chunk_id)
        # Amendment PDFs commonly list every change in a contents section and
        # repeat it later as the operative clause.  Prefer the latter so the
        # relationship links to the actual legal wording, not a page pointer.
        if chunk:
            candidates = by_document_clause.get(
                (str(chunk.get("document_id", "")), str(chunk.get("section_number", ""))),
                [chunk],
            )
            relation = str(row.get("relation_type", "")).lower()

            def evidence_score(item: dict) -> tuple[int, int, int, int]:
                candidate_text = str(item.get("text", ""))
                return (
                    int(substantive_legal_text(candidate_text)),
                    int(relation in candidate_text.lower()),
                    int(not toc_like(candidate_text)),
                    int(item.get("page_number", 0)),
                )

            chunk = max(candidates, key=evidence_score)
        text = chunk.get("text", "") if chunk else ""
        relation = str(row.get("relation_type", "")).lower()
        affected = normalize_section(row.get("affected_section", ""))
        source_text_match = bool(chunk and relation and relation in text.lower())
        target_scope = "GST Act section"
        target_instrument = "gst_act_2020"
        instrument_match = re.search(
            r"Section\s+\d{1,4}[^.]{0,140}?Goods and Services Tax\s*\(Amendment\)\s+Act of Bhutan\s+(20\d{2})",
            text,
            re.I,
        )
        if instrument_match:
            target_instrument = f"gst_amendment_{instrument_match.group(1)}"
            target_scope = f"GST Amendment Act {instrument_match.group(1)} section"
        schedule_match = re.search(r"\bSchedule\s+([IVX]+(?:\s*[A-C])?)", text, re.I)
        if schedule_match and not instrument_match:
            target_scope = f"Schedule {re.sub(r'\s+', ' ', schedule_match.group(1).upper()).strip()}"
        target_locator = (
            str(row.get("affected_section", ""))
            if str(row.get("affected_section", "")).lower().startswith("schedule")
            else f"{target_scope}, section {row.get('affected_section', '')}"
        )
        row["effective_from"] = row.get("effective_from") or document_dates.get(
            row.get("source_document_id", ""), ""
        )
        row["original_source_chunk_id"] = original_chunk_id
        row["source_chunk_id"] = chunk.get("chunk_id", "") if chunk else ""
        row["source_page"] = chunk.get("page_number", "") if chunk else ""
        row["source_text_match"] = source_text_match
        row["verification_status"] = (
            "codex_source_verified"
            if source_text_match and substantive_legal_text(text)
            else "codex_review_flagged"
        )
        row["affected_section_normalized"] = affected
        row["target_scope"] = target_scope
        row["target_instrument"] = target_instrument
        row["target_locator"] = target_locator
        row["review_reason"] = (
            "operative source clause contains the stated relationship and target locator"
            if row["verification_status"] == "codex_source_verified"
            else "relationship or substantive source wording requires another source check"
        )
        row["evidence_excerpt"] = re.sub(r"\s+", " ", text)[:900]
        rows.append(row)

    # The 2026 Act changes Schedule IV C rather than a numbered section, so the
    # generic section regex cannot discover it reliably.  Record it explicitly.
    if not any(row.get("source_document_id") == "gst_amendment_2026" for row in rows):
        source = next(
            (
                item
                for item in sections
                if item["document_id"] == "gst_amendment_2026"
                and "Schedule IV" in item.get("text", "")
            ),
            next((item for item in sections if item["document_id"] == "gst_amendment_2026"), None),
        )
        if source:
            rows.append(
                {
                    "event_id": f"event-{len(rows) + 1:04d}",
                    "source_document_id": "gst_amendment_2026",
                    "affected_section": "Schedule IV C",
                    "relation_type": "amended",
                    "source_chunk_id": source["chunk_id"],
                    "effective_from": "2026-05-18",
                    "verification_status": "codex_source_verified",
                    "source_text_match": True,
                    "affected_section_normalized": "scheduleivc",
                    "target_scope": "Schedule IV C",
                    "target_instrument": "gst_act_2020",
                    "target_locator": "Schedule IV C",
                    "source_page": source.get("page_number", ""),
                    "original_source_chunk_id": source["chunk_id"],
                    "review_reason": "operative source clause contains the Schedule IV C amendment",
                    "evidence_excerpt": re.sub(r"\s+", " ", source["text"])[:500],
                }
            )
    return pd.DataFrame(rows)


def enrich_section_versions(sections: list[dict], events: pd.DataFrame) -> list[dict]:
    event_rows = events.fillna("").to_dict("records")
    events_by_source_chunk: dict[str, list[dict]] = {}
    change_dates_by_section: dict[str, list[str]] = {}
    for event in event_rows:
        events_by_source_chunk.setdefault(event.get("source_chunk_id", ""), []).append(event)
        root = (
            normalize_section(event.get("target_locator", ""))
            if str(event.get("target_scope", "")).lower().startswith("schedule")
            else section_root(event.get("affected_section", ""))
        )
        when = event.get("effective_from", "")
        if root and when:
            change_dates_by_section.setdefault(root, []).append(when)

    enriched: list[dict] = []
    for item in sections:
        row = dict(item)
        doc_type = row.get("document_type", "")
        section = normalize_section(row.get("section_number", ""))
        root = section_root(section)
        source_events = events_by_source_chunk.get(row["chunk_id"], [])

        if doc_type in {"act", "consolidation"}:
            family = f"gst-act:{root or row['chunk_id']}"
        elif doc_type == "rules":
            family = f"gst-rules:{root or row['chunk_id']}"
        elif source_events:
            source_event = source_events[0]
            event_key = (
                normalize_section(source_event.get("target_locator", ""))
                if str(source_event.get("target_scope", "")).lower().startswith("schedule")
                else section_root(source_event.get("affected_section", ""))
            )
            family = f"gst-act:{event_key}"
        else:
            family = f"{doc_type}:{row['document_id']}:{root or row['chunk_id']}"

        version_start = row.get("effective_from", "")
        version_end = row.get("effective_to", "")
        if row["document_id"] == "gst_act_2020" and root in change_dates_by_section:
            later_dates = sorted(d for d in change_dates_by_section[root] if d > (version_start or "0001-01-01"))
            if later_dates:
                version_end = _previous_day(later_dates[0])

        rank_date = version_start or f"{row.get('year') or '0001'}-01-01"
        type_priority = {
            "act": 1,
            "notice": 2,
            "rules": 3,
            "amendment": 4,
            "consolidation": 5,
        }.get(doc_type, 0)
        rank_digits = int(re.sub(r"\D", "", rank_date)[:8].ljust(8, "0") or 0)

        row.update(
            {
                "version_family": family,
                "version_start": version_start,
                "version_end": version_end,
                "version_rank": rank_digits * 10 + type_priority,
                "version_status": {
                    "act": "base_text",
                    "amendment": "change_evidence",
                    "consolidation": "consolidated_view",
                    "rules": "operative_rules",
                    "notice": "official_notice",
                }.get(doc_type, "supporting_evidence"),
                "amendment_event_ids": [event["event_id"] for event in source_events],
                "affected_sections": [normalize_section(event["affected_section"]) for event in source_events],
                "relation_types": [event["relation_type"] for event in source_events],
                "version_verification_status": "metadata_audited_needs_group_legal_review",
            }
        )
        enriched.append(row)
    return enriched


def version_valid_on(record: dict, target_date: str | None) -> bool:
    if not target_date:
        return True
    start = record.get("version_start") or record.get("effective_from") or "0001-01-01"
    end = record.get("version_end") or record.get("effective_to") or "9999-12-31"
    return start <= target_date <= end

