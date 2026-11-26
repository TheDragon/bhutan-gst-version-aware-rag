from __future__ import annotations

import json
import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gst_rag.gemini import GeminiClient, gemini_from_env, load_env_file, rewrite_query
from gst_rag.paths import INDEXES, MANIFEST, METADATA, PROCESSED, RESULTS
from gst_rag.quality import (
    bm25_tokenize,
    gst_definition_intent,
    gst_question_has_evidence,
    gst_tax_needs_clarification,
    normalize_user_question,
    tobacco_context,
    vehicle_context,
)
from gst_rag.utils import read_jsonl

if TYPE_CHECKING:
    from gst_rag.generation import EvidenceGenerator
    from gst_rag.systems import RetrievalEngine
    from gst_rag.version_view import VersionIndex


APP_ROOT = Path(__file__).resolve().parents[2]
load_env_file(APP_ROOT / ".env")
GENERATOR_MODEL = os.getenv("GST_GENERATOR_MODEL", "google/flan-t5-base")
DEFAULT_TOP_K = 5

app = FastAPI(
    title="GST Lens API",
    version="0.1.0",
    description="Evidence-first, version-aware retrieval over official Bhutan GST documents.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SystemName = Literal["standard", "date_filtered", "version_aware"]


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    target_date: date | None = None
    system: SystemName = "version_aware"
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    target_date: date | None = None
    top_k: int = Field(default=3, ge=1, le=5)
    generate_answers: bool = False


@lru_cache(maxsize=1)
def get_engine() -> "RetrievalEngine":
    from gst_rag.systems import RetrievalEngine

    return RetrievalEngine.from_disk(INDEXES)


@lru_cache(maxsize=1)
def get_llm() -> GeminiClient | None:
    return gemini_from_env()


@lru_cache(maxsize=1)
def get_generator() -> "EvidenceGenerator":
    from gst_rag.generation import EvidenceGenerator

    return EvidenceGenerator(model_name=GENERATOR_MODEL, llm=get_llm())


@lru_cache(maxsize=1)
def get_version_index() -> "VersionIndex":
    from gst_rag.version_view import VersionIndex

    return VersionIndex(get_chunks())


@lru_cache(maxsize=512)
def _understand(question: str, today: str) -> dict:
    """Gemini's cleaned question, time intent and search terms ({} when Gemini is off)."""
    llm = get_llm()
    return (rewrite_query(llm, question, today) or {}) if llm else {}


def _chunk_by_id(chunk_id: str) -> dict | None:
    return get_version_index().by_id.get(chunk_id)


@lru_cache(maxsize=1)
def get_chunks() -> list[dict]:
    return read_jsonl(PROCESSED / "sections.jsonl")


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _clean_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6)


def _evidence_item(item: dict, rank: int, citations: list[dict] | None = None) -> dict:
    return {
        "rank": rank,
        "cited": any(c.get("chunk_id") == item.get("chunk_id") for c in citations or []),
        "chunk_id": item.get("chunk_id", ""),
        "document_id": item.get("document_id", ""),
        "title": item.get("title", ""),
        "document_type": item.get("document_type", ""),
        "section_number": str(item.get("section_number", "")),
        "heading": item.get("heading", ""),
        "page_number": item.get("page_number"),
        "text": item.get("text", ""),
        "source_url": item.get("source_url", ""),
        "effective_from": item.get("effective_from", ""),
        "effective_to": item.get("effective_to", ""),
        "version_start": item.get("version_start", ""),
        "version_end": item.get("version_end", ""),
        "version_status": item.get("version_status", ""),
        "version_family": item.get("version_family", ""),
        "affected_sections": item.get("affected_sections", []),
        "relation_types": item.get("relation_types", []),
        "requires_manual_review": bool(item.get("requires_manual_review", False)),
        "scores": {
            "rrf": _clean_score(item.get("rrf_score")),
            "semantic": _clean_score(item.get("dense_score")),
            "bm25": _clean_score(item.get("bm25_score")),
        },
    }


def _retrieve(request: SearchRequest, expansion: str = "") -> tuple[str | None, list[dict]]:
    target_date = _iso(request.target_date)
    results = get_engine().search(
        query=request.question,
        system=request.system,
        target_date=target_date,
        top_k=request.top_k,
        expansion=expansion,
    )
    resolved_date = results[0].get("query_date") if results else target_date
    return resolved_date, results


@app.get("/api/health")
def health() -> dict:
    metadata = json.loads((INDEXES / "index_metadata.json").read_text(encoding="utf-8"))
    return {
        "status": "ready",
        "service": "GST Lens API",
        "embedding_model": metadata.get("model"),
        "generator_model": GENERATOR_MODEL,
        "llm_model": get_llm().model if get_llm() else None,
        "indexed_chunks": len(metadata.get("chunks", [])),
        "evaluation_status": "provisional_until_group_manual_verification",
    }


@app.get("/api/stats")
def stats() -> dict:
    manifest = pd.read_csv(MANIFEST).fillna("")
    events = pd.read_csv(METADATA / "amendment_events.csv").fillna("")
    metrics_path = RESULTS / "commencement_evaluation" / "controlled_comparison_metrics.json"
    if not metrics_path.exists():
        metrics_path = RESULTS / "controlled_comparison_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "documents": int(len(manifest)),
        "authoritative_documents": int((manifest["availability_status"] == "available").sum()),
        "chunks": len(get_chunks()),
        "version_events": int(len(events)),
        "benchmark_questions": 120,
        "evaluation_status": "provisional",
        "systems": metrics.get("systems", metrics),
    }


@app.get("/api/documents")
def documents() -> list[dict]:
    manifest = pd.read_csv(MANIFEST).fillna("")
    chunk_counts: dict[str, int] = {}
    for item in get_chunks():
        doc_id = item.get("document_id", "")
        chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
    records = []
    for row in manifest.to_dict(orient="records"):
        row["chunk_count"] = chunk_counts.get(row["document_id"], 0)
        row["requires_manual_review"] = str(row["requires_manual_review"]).lower() == "true"
        records.append(row)
    return records


@app.get("/api/version-events")
def version_events(
    section: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=100, ge=1, le=250),
) -> list[dict]:
    events = pd.read_csv(METADATA / "amendment_events.csv").fillna("")
    if section:
        normalized = re.sub(r"\s+", "", section).lower()
        events = events[
            events["affected_section_normalized"].astype(str).str.replace(" ", "").str.lower() == normalized
        ]
    return events.head(limit).to_dict(orient="records")


@app.get("/api/timeline/{section}")
def timeline(section: str) -> dict:
    normalized = re.sub(r"\s+", "", section).lower()
    matches = []
    for item in get_chunks():
        section_number = re.sub(r"\s+", "", str(item.get("section_number", ""))).lower()
        affected = [re.sub(r"\s+", "", str(value)).lower() for value in item.get("affected_sections", [])]
        if normalized == section_number or normalized in affected:
            matches.append(item)
    matches.sort(key=lambda item: (str(item.get("version_start", "")), int(item.get("version_rank", 0))))
    return {
        "section": section,
        "count": len(matches),
        "entries": [_evidence_item(item, rank) for rank, item in enumerate(matches, start=1)],
    }


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    resolved_date, results = _retrieve(request)
    return {
        "question": request.question,
        "system": request.system,
        "target_date": resolved_date,
        "count": len(results),
        "evidence": [_evidence_item(item, rank) for rank, item in enumerate(results, start=1)],
    }


@app.post("/api/answer")
def answer(request: SearchRequest) -> dict:
    original_question = request.question
    question = normalize_user_question(original_question)
    response = {"question": original_question, "system": request.system}
    if question != original_question:
        response["interpreted_question"] = question
    clarification = None
    if gst_tax_needs_clarification(question):
        gst_question = None
        if not re.search(r"\b(?:gst|excise|duty|duties|total tax(?:es)?|all taxes)\b", question, re.I):
            gst_question = re.sub(r"\btax(es)?\b", "GST", question, count=1, flags=re.I)
        message = (
            "GST and tobacco excise are separate taxes. I can look up GST in the indexed sources."
            if tobacco_context(question)
            else "GST is one part of what you might pay on a vehicle. I can look up GST in the indexed sources."
        )
        clarification = {
            "clarification_title": "Which tax do you mean?",
            "answer": message,
            "clarification_question": gst_question,
            "clarification_action": "Search GST only",
            "reason": "clarify_tax_type",
        }
    else:
        words = " ".join(bm25_tokenize(question))
        if words in {"tax", "tax please", "taxes", "taxes please"}:
            clarification = {"clarification_title": "What would you like to know about GST?", "answer": "Try asking about the GST rate, a product, or when to file a GST return."}
        elif words in {"exemption", "exemptions", "gst exemption", "gst exemptions"}:
            clarification = {"clarification_title": "What are you checking for an exemption?", "answer": "Name the goods or services, and tell me whether they’re sold in Bhutan or imported."}
        elif re.fullmatch(r"section\s+\d+[a-z]?(?:\s*\([^)]+\))?[?.]?", question, re.I):
            clarification = {"clarification_title": "Which document is the section from?", "answer": "The GST Act and the GST Rules can use the same number. Name the document and tell me what you’d like to know about the section."}

    if clarification is not None:
        return {
            **response,
            "target_date": _iso(request.target_date),
            "reason": "clarify_question",
            "abstained": True,
            "citations": [],
            "evidence": [],
            **clarification,
        }

    from gst_rag.generation import lookup_gst_definition, lookup_practical_gst_answer, present_answer

    today = _today()
    understanding = _understand(question, today)
    search_terms = understanding.get("search_terms", "")
    # Hand-written lookups match exact phrasings, so give them the spelling-fixed question.
    lookup_question = understanding.get("clean_question") or question

    time_note = ""
    if request.target_date is None:
        resolved_date, time_note = _resolve_time(request.model_copy(update={"question": question}), understanding, today)
        request = request.model_copy(update={"target_date": date.fromisoformat(resolved_date)})
    request = request.model_copy(update={"question": question})
    resolved_date = _iso(request.target_date)

    direct_answer = None
    if gst_definition_intent(question) or gst_definition_intent(lookup_question):
        direct_answer = lookup_gst_definition(get_chunks(), resolved_date, request.top_k)
    elif re.search(r"\b(register|registration|rice)\b", question, re.I):
        direct_answer = lookup_practical_gst_answer(question, get_chunks(), resolved_date, request.top_k) or (
            lookup_practical_gst_answer(lookup_question, get_chunks(), resolved_date, request.top_k)
            if lookup_question != question else None
        )

    related: list[dict] = []
    if direct_answer is None:
        resolved_date, results = _retrieve(request, search_terms)
        section_lookup = re.fullmatch(
            r"what does section (\d+[a-z]?) of (?:the )?(consolidated gst act|gst rules(?: 2026)?) say\?*",
            question,
            re.I,
        )
        if section_lookup:
            section, instrument = section_lookup.groups()
            document_id = "gst_consolidated_2025" if instrument.lower().startswith("consolidated") else "gst_rules_2026"
            results = [
                item for item in results
                if item["document_id"] == document_id and item["section_number"].lower() == section.lower()
            ]
        section_topic = re.fullmatch(r"what does section (\d+[a-z]?) say about ([^?]+)\?*", question, re.I)
        if section_topic:
            section, topic = section_topic.groups()
            terms = set(bm25_tokenize(topic)) - {"a", "an", "the", "gst"}
            results = [
                item for item in results
                if item["section_number"].lower() == section.lower()
                and terms and terms <= set(bm25_tokenize(item["text"]))
            ]
        exemption = re.match(r"\b(?:is|are) (?:imported )?(.+?) (?:imports? )?exempt from gst\b", question, re.I)
        if exemption:
            product = set(bm25_tokenize(exemption[1])) - {"a", "an", "the"}
            narrowed = [
                item for item in results
                if product and product <= set(bm25_tokenize(item["text"]))
                and re.search(r"\bexempt\w*\b", item["text"], re.I)
            ]
            # Schedules name products by trade name ("soya-bean oil", not
            # "cooking oil"); if the narrow match finds nothing, keep the
            # ranked results and let the evidence check decide.
            results = narrowed or results
        product_tax = re.fullmatch(
            r"(?:what is the )?(?:gst|tax)(?: rate)? (?:on|for) (?:imported )?(.+?)"
            r"(?: in bhutan)?(?: latest| today| as of now)?[?.]?",
            question,
            re.I,
        )
        if product_tax:
            product = set(bm25_tokenize(product_tax[1])) - {"a", "an", "the"}
            if product and not (tobacco_context(question) or vehicle_context(question)):
                results = [item for item in results if product <= set(bm25_tokenize(item["text"]))] or results
        # The keyword check below rejects casual wording ("is butter tax free
        # now") unless every word sits in one passage. When Gemini is on it
        # reads the passages itself, says when they don't answer the question,
        # and suggests a better one, so the check is only needed offline.
        if not results or (get_llm() is None and not gst_question_has_evidence(request.question, results)):
            generated = {
                "answer": "I couldn’t find an official GST passage about that. Try naming the GST rule, product, or transaction you’re asking about.",
                "abstained": True,
                "reason": "no_topic_match",
                "citations": [],
            }
            results = []
        else:
            from gst_rag.quality import best_answer_evidence

            # Show the answer model the other version of the main passage so it
            # can say what changed; it may not cite it.
            related = get_version_index().related_passages(best_answer_evidence(results), resolved_date)
            generated = get_generator().answer(request.question, results, resolved_date, related)
    else:
        generated, results = direct_answer
        generated["generator"] = GENERATOR_MODEL
    generated = present_answer(question, generated, results)

    version_context = None
    citations = generated.get("citations") or []
    if not generated.get("abstained") and citations:
        from gst_rag.version_view import compare_covers_quote

        version_context = get_version_index().context(_chunk_by_id(citations[0]["chunk_id"]), resolved_date)
        if version_context and version_context.get("compare") and not compare_covers_quote(
            version_context, generated.get("evidence_quote", "")
        ):
            version_context["compare"] = None
        compare = (version_context or {}).get("compare")
        shown = {compare["before"]["chunk_id"], compare["after"]["chunk_id"]} if compare else set()
        # Gemini's change summary describes the version pair it was shown;
        # drop it if the cited passage ended up being a different one.
        if not related or related[0]["chunk_id"] not in shown:
            generated.pop("change_summary", None)
    else:
        generated.pop("change_summary", None)
    suggestion = generated.pop("suggestion", "")
    return {
        **response,
        "target_date": resolved_date,
        **({"time_note": time_note} if time_note else {}),
        **generated,
        **({"suggestion": suggestion} if suggestion else {}),
        **({"search_expansion": search_terms} if search_terms else {}),
        "version_context": version_context,
        "evidence": [_evidence_item(item, rank, citations) for rank, item in enumerate(results, start=1)],
        "notice": "Research prototype. Verify the cited official source before relying on the answer.",
    }


def _today() -> str:
    """Today's date in Bhutan (UTC+6), the default legal date for the chat."""
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone(timedelta(hours=6))).date().isoformat()


def _resolve_time(request: SearchRequest, understanding: dict, today: str) -> tuple[str, str]:
    """Work out which date the question is about when the user gave none.

    Returns (legal date, note shown above the answer or "").
    """
    from gst_rag.generation import readable_date
    from gst_rag.quality import best_answer_evidence
    from gst_rag.retrieval import parse_query_date

    intent = understanding.get("time_intent", "current")
    if understanding:
        named = understanding.get("time_date") if intent == "date" else ""
    else:
        # Offline: only a full date counts; a year alone is usually part of a
        # document name ("GST Act 2020"), not the date being asked about.
        # parse_query_date reads ISO dates only (it maps a bare year to 31 Dec).
        full_date = re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", request.question)
        named = parse_query_date(request.question) if full_date else ""
    if named and named <= today:
        return named, f"Answering for {readable_date(named)}, the date in your question."
    if intent != "previous" and not re.search(r"\b(before the (?:latest |last |recent )?(?:change|amendment)|previously|used to be|originally)\b", request.question, re.I):
        return today, ""
    # "What was it before the change?": find the provision as it is today,
    # then step back to the day before its current version took effect.
    probe = request.model_copy(update={"target_date": date.fromisoformat(today)})
    _, results = _retrieve(probe, understanding.get("search_terms", ""))
    if not results:
        return today, ""
    ctx = get_version_index().context(best_answer_evidence(results), today)
    timeline = (ctx or {}).get("timeline") or []
    position = next((i for i, row in enumerate(timeline) if row["selected"]), None)
    if not position:
        return today, "I couldn’t find an earlier version of this rule, so this is the current one."
    earlier, current = timeline[position - 1], timeline[position]
    when = earlier["end"] or current["start"]
    return when, f"Answering for {readable_date(when)}, before the change that took effect on {readable_date(current['start'])}."


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict:
    comparisons = []
    for system in ("standard", "date_filtered", "version_aware"):
        search_request = SearchRequest(
            question=request.question,
            target_date=request.target_date,
            system=system,
            top_k=request.top_k,
        )
        resolved_date, results = _retrieve(search_request)
        payload = {
            "system": system,
            "target_date": resolved_date,
            "evidence": [_evidence_item(item, rank) for rank, item in enumerate(results, start=1)],
        }
        if request.generate_answers:
            payload["generation"] = get_generator().answer(request.question, results, resolved_date)
        comparisons.append(payload)
    return {"question": request.question, "comparisons": comparisons}


@app.get("/api/chunks/{chunk_id:path}")
def chunk_detail(chunk_id: str) -> dict:
    item = next((chunk for chunk in get_chunks() if chunk.get("chunk_id") == chunk_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return _evidence_item(item, 1)


@app.get("/")
def root() -> dict:
    return {"name": "GST Lens API", "docs": "/docs", "health": "/api/health"}
