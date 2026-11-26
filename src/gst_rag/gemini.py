"""Optional Google Gemini support for GST Lens.

Gemini is used in two places, and only when ``GEMINI_API_KEY`` is set:

1. ``rewrite_query`` adds statutory search terms to a plain-language question
   before retrieval (for example, "is rice taxed" -> "rice exempt supply
   Schedule ... GST rate").  The original question is still used for date,
   section and amendment logic; the rewrite only widens the keyword and
   embedding search.
2. ``grounded_answer`` writes the answer from the retrieved passages.  The model
   must return the ids of the passages it used and a verbatim quote.  Both are
   checked in code, so an answer can only cite passages the retriever returned.

Any failure (no key, network error, bad JSON, failed checks) returns ``None`` so
callers fall back to the offline extractive generator.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.7-flash"
PASSAGE_CHARS = 2500


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass
class GeminiClient:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout: float = 30.0
    retries: int = 2

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        url = f"{API_ROOT}/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for attempt in range(self.retries + 1):
            response = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code != 200:
                detail = response.text[:300].replace(self.api_key, "***")
                raise RuntimeError(f"Gemini HTTP {response.status_code}: {detail}")
            break
        payload = response.json()
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        return json.loads(text)


def gemini_from_env() -> GeminiClient | None:
    if os.getenv("GST_USE_GEMINI", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    return GeminiClient(api_key=key, model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = (
    "You help search Bhutan's GST legislation (GST Act 2020, amendments, consolidated Act, GST Rules, "
    "and implementation notices). Return four things.\n"
    "clean_question: the user's question with spelling fixed, as one clear question in plain English. "
    "Keep every product, service, section and time reference. Do not add facts or a date.\n"
    "time_intent: 'previous' if the user asks how the rule was before the latest change, previously, "
    "originally, or what it used to be; 'date' if they name a specific date, month or year; otherwise "
    "'current'.\n"
    "time_date: only for time_intent 'date': that date as YYYY-MM-DD (a year alone means 31 December of "
    "that year; 'before <date>' means the day before). Otherwise empty.\n"
    "search_terms: keyword search terms that would appear in the law. Keep every product, service, "
    "person or section named. Add statutory wording such as: taxable supply, taxable import, exempt "
    "supply, exempt import, zero-rated, Schedule, GST rate, input tax credit, registration, GST return, "
    "tax period, as relevant. For a question about whether a product is taxed, include both 'exempt' "
    "and 'GST rate' terms and the product's common names. Do not add dates, years, or the words "
    "amend/amended/repeal/insert/change. At most 25 words."
)
REWRITE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "clean_question": {"type": "STRING"},
        "time_intent": {"type": "STRING", "enum": ["current", "previous", "date"]},
        "time_date": {"type": "STRING"},
        "search_terms": {"type": "STRING"},
    },
    "required": ["clean_question", "time_intent", "search_terms"],
}
_BANNED_REWRITE = re.compile(r"\b(?:19|20)\d{2}\b|\b(amend\w*|repeal\w*|insert\w*|replace\w*|chang\w*)\b", re.I)


def rewrite_query(client: GeminiClient, question: str, today: str | None = None) -> dict | None:
    """Return {"clean_question", "time_intent", "time_date", "search_terms"}, or None on failure."""
    from datetime import date

    prompt = f"Today's date: {today or date.today().isoformat()}\nQuestion: {question}"
    try:
        data = client.generate_json(REWRITE_SYSTEM, prompt, REWRITE_SCHEMA)
    except Exception as exc:  # network, quota, JSON
        log.warning("Gemini query rewrite failed: %s", exc)
        return None
    terms = " ".join(str(data.get("search_terms", "")).split())
    # The original question drives date and amendment filtering; keep the
    # expansion from switching those modes on by accident.
    terms = " ".join(_BANNED_REWRITE.sub(" ", terms).split())
    clean = " ".join(str(data.get("clean_question", "")).split())
    intent = str(data.get("time_intent", "current")).strip().lower()
    when = str(data.get("time_date", "")).strip()
    try:
        when = date.fromisoformat(when).isoformat() if intent == "date" else ""
    except ValueError:
        when = ""
    if intent == "date" and not when:
        intent = "current"
    if intent not in {"current", "previous", "date"}:
        intent = "current"
    return {
        "search_terms": terms if len(terms) <= 300 else "",
        "clean_question": clean if 3 <= len(clean) <= 400 else "",
        "time_intent": intent,
        "time_date": when,
    }


# ---------------------------------------------------------------------------
# Grounded answers
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You answer questions about Bhutan's GST law for ordinary people and small businesses. "
    "Use ONLY the numbered passages provided; never use outside knowledge. "
    "Write 1-4 short sentences in plain English, second person where natural. "
    "Answer the actual question first: for a yes/no question start with Yes, No, or It depends. "
    "Include conditions, time limits or rates the passages state. "
    "If the passages give a general rule but do not mention the specific product, service or situation "
    "the user asked about, give the general rule and say plainly that the passages shown do not mention "
    "that item specifically, so an exemption or special rule cannot be ruled out. "
    "If the passages do not address the question at all, set answerable to false. "
    "supporting_ids must list the ids (like E1) of the passages your answer relies on, most important first. "
    "quote must be copied exactly, word for word, from the first supporting passage: the single clause or "
    "sentence that best supports the answer, under 60 words.\n"
    "Some passages may be labelled EARLIER VERSION or LATER VERSION (ids like V1). They are NOT in force on "
    "the legal date. Never answer from them, and never list them in supporting_ids. Use them only for "
    "change_summary: one plain sentence saying exactly what differs between that version and the version "
    "in force (for example 'Before 1 January 2026 the rate was 7%; it is now 5%.'). If the only difference "
    "is a removed reference or wording cleanup, say so plainly. Leave change_summary empty if there is no "
    "such passage or no real difference.\n"
    "answer_quality: 'complete' if the passages fully answer the question; 'partial' if you could only give "
    "a general rule or part of the answer (for example the item asked about is not mentioned); 'none' if "
    "they do not answer it.\n"
    "better_question: when answer_quality is not 'complete', one short question (under 15 words) that is "
    "closest to what the user wanted AND that the passages shown answer fully. Otherwise empty."
)
ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answerable": {"type": "BOOLEAN"},
        "answer": {"type": "STRING"},
        "supporting_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "quote": {"type": "STRING"},
        "change_summary": {"type": "STRING"},
        "answer_quality": {"type": "STRING", "enum": ["complete", "partial", "none"]},
        "better_question": {"type": "STRING"},
    },
    "required": ["answerable", "answer", "supporting_ids", "quote", "answer_quality"],
}


def _flat(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).lower()


def _passage_block(label: str, item: dict) -> str:
    start = item.get("version_start") or item.get("effective_from") or "unknown"
    end = item.get("version_end") or item.get("effective_to") or "open"
    text = " ".join(str(item.get("text", "")).split())[:PASSAGE_CHARS]
    version = f" | {item['_version_label']} (not in force on the legal date)" if item.get("_version_label") else ""
    return (
        f"[{label}] {item.get('title', '')} | section {item.get('section_number', '')} | "
        f"page {item.get('page_number', '')} | in force {start} to {end}{version}\n{text}"
    )


def _suggestion(data: dict) -> str:
    """The model's better question, only when its answer was not complete."""
    quality = str(data.get("answer_quality", "complete")).lower()
    if data.get("answerable") and quality == "complete":
        return ""
    text = " ".join(str(data.get("better_question", "")).split()).strip()
    if not 5 <= len(text) <= 140:
        return ""
    return text if text.endswith("?") else text + "?"


def grounded_answer(
    client: GeminiClient,
    question: str,
    evidence: list[dict],
    target_date: str | None,
    related: list[dict] | None = None,
) -> tuple[dict | None, str]:
    """Return (result, note). ``result`` is None when the caller should fall back.

    ``related`` holds other versions of the main passage (labelled with
    ``_version_label``). They are shown to the model only to explain what
    changed and can never be cited.
    """
    if not evidence:
        return None, "no evidence"
    blocks = [_passage_block(f"E{i}", item) for i, item in enumerate(evidence, start=1)]
    blocks += [_passage_block(f"V{i}", item) for i, item in enumerate(related or [], start=1)]
    passages = "\n\n".join(blocks)
    prompt = (
        f"Legal date the user selected: {target_date or 'not stated (use the latest passages)'}\n"
        f"Question: {question}\n\nPassages:\n{passages}"
    )
    try:
        data = client.generate_json(ANSWER_SYSTEM, prompt, ANSWER_SCHEMA)
    except Exception as exc:
        log.warning("Gemini answer failed: %s", exc)
        return None, str(exc)[:200]

    by_id = {f"E{i}": item for i, item in enumerate(evidence, start=1)}
    answer = " ".join(str(data.get("answer", "")).split())
    if not data.get("answerable"):
        return {
            "answer": "I couldn’t find the answer to that in the official GST texts.",
            "abstained": True,
            "reason": "llm_insufficient_evidence",
            "answer_strategy": "llm_grounded",
            "citations": [],
            "suggestion": _suggestion(data),
        }, ""

    ids = [str(value).strip().upper() for value in data.get("supporting_ids", [])]
    cited = [by_id[value] for value in dict.fromkeys(ids) if value in by_id]
    if not answer or not cited:
        return None, "answer did not cite a retrieved passage"

    quote = " ".join(str(data.get("quote", "")).split()).strip(' "“”')
    if quote and _flat(quote) not in _flat(str(cited[0].get("text", ""))):
        # Never show a "quote" that isn't in the source. present_answer will
        # pick an excerpt from the cited passage instead.
        quote = ""
    change = " ".join(str(data.get("change_summary", "")).split()) if related else ""
    return {
        "answer": answer,
        "abstained": False,
        "reason": "",
        "answer_strategy": "llm_grounded",
        "evidence_quote": quote,
        "cited_items": cited,
        "change_summary": change[:400],
        "suggestion": _suggestion(data),
    }, ""
