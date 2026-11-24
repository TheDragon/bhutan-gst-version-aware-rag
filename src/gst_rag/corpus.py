from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

from .paths import EXTRACTED, MANIFEST, METADATA, PROCESSED, RAW_PDFS, TRANSCRIPTIONS, ensure_directories
from .utils import normalize_text, sha256_file, write_jsonl


SECTION_PATTERN = re.compile(r"(?m)^(?:Section\s+)?(\d{1,4}[A-Za-z]?(?:\s*\([^)]+\))?)\.?\s+(.{8,})$")
EVENT_PATTERN = re.compile(
    r"(?is)(?:section|sections)\s+(\d{1,4}[A-Za-z]?(?:\s*\([^)]+\))?)"
    r".{0,180}?\b(amended|repealed|inserted|substituted|replaced)\b"
)


def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"document_id", "title", "document_type", "source_url", "local_filename", "availability_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame["document_id"].duplicated().any():
        raise ValueError("Manifest document_id values must be unique")
    return frame


def download_corpus(timeout: int = 90) -> pd.DataFrame:
    ensure_directories()
    manifest = load_manifest()
    rows: list[dict] = []
    headers = {"User-Agent": "GCIT-NLP-Research/1.0 (academic corpus collection)"}
    for row in manifest.to_dict("records"):
        if row["availability_status"] != "available" or not row["source_url"]:
            rows.append({**row, "download_status": "skipped", "sha256": "", "bytes": 0})
            continue
        target = RAW_PDFS / row["local_filename"]
        status = "existing"
        if not target.exists():
            response = requests.get(row["source_url"], headers=headers, timeout=timeout)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"Downloaded content is not a PDF: {row['source_url']}")
            target.write_bytes(response.content)
            status = "downloaded"
        if not target.read_bytes()[:4] == b"%PDF":
            raise ValueError(f"Invalid PDF signature: {target}")
        rows.append(
            {
                **row,
                "download_status": status,
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
                "download_checked_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(METADATA / "download_audit.csv", index=False, encoding="utf-8")
    return audit


def _is_probably_english(text: str) -> bool:
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    all_letters = sum(ch.isalpha() for ch in text)
    return ascii_letters >= 80 and ascii_letters / max(all_letters, 1) >= 0.55


def extract_corpus() -> pd.DataFrame:
    ensure_directories()
    manifest = load_manifest()
    reports: list[dict] = []
    for doc in manifest.to_dict("records"):
        pdf_path = RAW_PDFS / doc["local_filename"]
        if not pdf_path.exists():
            reports.append({"document_id": doc["document_id"], "status": "missing", "pages": 0, "english_pages": 0, "characters": 0})
            continue
        reader = PdfReader(str(pdf_path))
        transcription_path = TRANSCRIPTIONS / f"{doc['document_id']}.txt"
        manual_transcription = normalize_text(transcription_path.read_text(encoding="utf-8")) if transcription_path.exists() else ""
        pages: list[dict] = []
        errors: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = normalize_text(page.extract_text() or "")
            except Exception as exc:  # a corrupt page should not stop the corpus audit
                text = ""
                errors.append(f"page {index}: {type(exc).__name__}")
            extraction_method = "pdf_text"
            if not _is_probably_english(text) and manual_transcription and len(reader.pages) == 1:
                text = manual_transcription
                extraction_method = "manual_visual_transcription"
            pages.append(
                {
                    "document_id": doc["document_id"],
                    "page_number": index,
                    "text": text,
                    "language_candidate": "en" if _is_probably_english(text) else "other_or_empty",
                    "extraction_method": extraction_method,
                }
            )
        write_jsonl(EXTRACTED / f"{doc['document_id']}.jsonl", pages)
        english = [p for p in pages if p["language_candidate"] == "en"]
        reports.append(
            {
                "document_id": doc["document_id"],
                "status": ("extracted_with_manual_transcription" if manual_transcription and english else "extracted") if english else "needs_ocr_or_review",
                "pages": len(pages),
                "english_pages": len(english),
                "characters": sum(len(p["text"]) for p in english),
                "errors": "; ".join(errors),
            }
        )
    report = pd.DataFrame(reports)
    report.to_csv(METADATA / "extraction_report.csv", index=False, encoding="utf-8")
    return report


def _page_chunks(text: str) -> list[tuple[str, str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    chunks: list[tuple[str, str, str]] = []
    if matches:
        for idx, match in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = normalize_text(text[match.start():end])
            if len(body.split()) >= 18:
                chunks.append((match.group(1).replace(" ", ""), match.group(2).strip()[:180], body))
    if not chunks:
        paragraphs = [normalize_text(p) for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 25]
        for idx, paragraph in enumerate(paragraphs):
            chunks.append((f"page-part-{idx + 1}", paragraph.split("\n", 1)[0][:180], paragraph))
    return chunks


def build_sections() -> list[dict]:
    manifest = load_manifest().set_index("document_id").to_dict("index")
    records: list[dict] = []
    for path in sorted(EXTRACTED.glob("*.jsonl")):
        doc_id = path.stem
        if doc_id not in manifest:
            continue
        with path.open("r", encoding="utf-8") as stream:
            import json

            pages = [json.loads(line) for line in stream if line.strip()]
        doc = manifest[doc_id]
        for page in pages:
            if page["language_candidate"] != "en":
                continue
            for part_index, (section_number, heading, text) in enumerate(_page_chunks(page["text"]), start=1):
                chunk_id = f"{doc_id}:p{page['page_number']}:s{section_number}:{part_index}"
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "title": doc["title"],
                        "document_type": doc["document_type"],
                        "year": doc["year"],
                        "section_number": section_number,
                        "heading": heading,
                        "page_number": page["page_number"],
                        "text": text,
                        "effective_from": doc["effective_from"],
                        "effective_to": doc["effective_to"],
                        "source_url": doc["source_url"],
                        "official_status": doc["official_status"],
                        "requires_manual_review": True,
                    }
                )
    write_jsonl(PROCESSED / "sections.jsonl", records)
    return records


def extract_amendment_events(sections: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in sections:
        if chunk["document_type"] not in {"amendment", "notice"}:
            continue
        for match in EVENT_PATTERN.finditer(chunk["text"]):
            key = (chunk["document_id"], match.group(1).replace(" ", ""), match.group(2).lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "event_id": f"event-{len(rows) + 1:04d}",
                    "source_document_id": chunk["document_id"],
                    "affected_section": key[1],
                    "relation_type": key[2],
                    "source_chunk_id": chunk["chunk_id"],
                    "effective_from": chunk["effective_from"],
                    "verification_status": "needs_manual_review",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(METADATA / "amendment_events.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return frame


def corpus_summary(sections: list[dict]) -> dict:
    return {
        "documents": len({r["document_id"] for r in sections}),
        "chunks": len(sections),
        "document_types": dict(Counter(r["document_type"] for r in sections)),
        "manual_review_required": sum(bool(r["requires_manual_review"]) for r in sections),
    }
