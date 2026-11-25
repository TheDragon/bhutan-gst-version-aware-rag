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

from gst_rag.paths import INDEXES, MANIFEST, METADATA, PROCESSED, RESULTS
from gst_rag.utils import read_jsonl

if TYPE_CHECKING:
    from gst_rag.generation import EvidenceGenerator
    from gst_rag.systems import RetrievalEngine


APP_ROOT = Path(__file__).resolve().parents[2]
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
def get_generator() -> "EvidenceGenerator":
    from gst_rag.generation import EvidenceGenerator

    return EvidenceGenerator(model_name=GENERATOR_MODEL)


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


def _evidence_item(item: dict, rank: int) -> dict:
    return {
        "rank": rank,
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


def _retrieve(request: SearchRequest) -> tuple[str | None, list[dict]]:
    target_date = _iso(request.target_date)
    results = get_engine().search(
        query=request.question,
        system=request.system,
        target_date=target_date,
        top_k=request.top_k,
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
    resolved_date, results = _retrieve(request)
    generated = get_generator().answer(request.question, results, resolved_date)
    return {
        "question": request.question,
        "system": request.system,
        "target_date": resolved_date,
        **generated,
        "evidence": [_evidence_item(item, rank) for rank, item in enumerate(results, start=1)],
        "notice": "Research prototype. Verify the cited official source before relying on the answer.",
    }


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
