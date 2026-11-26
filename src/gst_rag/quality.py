from __future__ import annotations

import re


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_user_question(question: str) -> str:
    """Normalize known spelling variants without rewriting legal references."""
    question = " ".join(question.split())
    question = re.sub(
        r"\b(?:cigarrete|ciggarrette|cigaret)(s?)\b",
        lambda match: "cigarette" + match.group(1).lower(),
        question,
        flags=re.I,
    )
    words = " ".join(bm25_tokenize(question))
    if words in {"gst return deadline", "gst filing deadline", "gst return due date"}:
        return "When must a taxable person file a GST return after the end of each tax period?"
    return question


def gst_tax_needs_clarification(question: str) -> bool:
    if not (tobacco_context(question) or vehicle_context(question)):
        return False
    gst = re.search(r"\b(gst|goods and services tax)\b", question, re.I)
    return bool(
        re.search(r"\b(excise|duty|duties|total tax(?:es)?|all taxes)\b", question, re.I)
        or (not gst and re.search(r"\btax(?:es)?\b", question, re.I))
    )


def gst_definition_intent(question: str) -> bool:
    words = " ".join(bm25_tokenize(question))
    term = r"(?:gst|goods and services tax)"
    return bool(re.fullmatch(
        rf"(?:{term}|what (?:is|s) (?:the )?{term}|what does {term} (?:mean|stand for)|"
        rf"define (?:the )?{term}|what is the (?:meaning|definition) of {term})"
        r"(?: in bhutan)?(?: please)?",
        words,
    ))


def commencement_intent(query: str) -> bool:
    if re.search(r"\b(commencement|effective date)\b", query, re.I):
        return True
    return bool(
        re.search(r"\b(when|what date|on which date)\b", query, re.I)
        and re.search(r"\b(force|effect|effective|commenc\w*|start\w*|appl\w*)\b", query, re.I)
    )


def act_commencement_clause(item: dict) -> str:
    """Extract an Act-level commencement clause from operative text."""
    raw = str(item.get("text", ""))
    if item.get("document_type") not in {"act", "amendment", "consolidation"} or toc_like(raw):
        return ""
    match = re.search(
        r"\bThis Act\s+(?:(?:shall\s+)?come[s]? into force|commences|takes effect)\b[^.]*\.",
        " ".join(raw.split()),
        re.I,
    )
    return match.group() if match else ""


def gst_rate_intent(query: str) -> bool:
    if commencement_intent(query):
        return False
    lowered = query.lower()
    has_gst = "gst" in lowered or "goods and services tax" in lowered
    asks_rate = bool(re.search(r"\b(rate|percent|percentage|how much|payable|appl(?:y|ies|ied)|charg(?:e|ed))\b", lowered))
    tobacco_rate = tobacco_context(lowered) and re.search(
        r"\b(?:gst|goods and services tax)\s+(?:on|for|tobacco|cigarettes?|cigars?)\b|"
        r"\b(?:tobacco(?: products)?|cigarettes?|cigars?)\s+gst\b",
        lowered,
    )
    return has_gst and bool(asks_rate or vehicle_context(lowered) or tobacco_rate)


def tobacco_context(query: str) -> bool:
    return bool(re.search(r"\b(tobacco|cigarettes?|cigars?)\b", query, re.I))


def vehicle_context(query: str) -> bool:
    return bool(re.search(r"\b(cars?|vehicles?|automobile|motorcar)\b", query, re.I))


def gst_question_has_evidence(question: str, evidence: list[dict]) -> bool:
    """Require a topical match for vague queries and unfamiliar GST rate subjects."""
    rate_subject = re.search(r"\bgst rate\s+(?:on|for)\s+([^?.]+)", question, re.I)
    if rate_subject and not (tobacco_context(question) or vehicle_context(question)):
        phrase = re.split(r"\b(?:as of|in bhutan|today|currently)\b", rate_subject.group(1), maxsplit=1, flags=re.I)[0]
    elif re.search(r"\b(gst|goods and services tax|section|schedule|amendment)\b", question, re.I):
        return True
    else:
        phrase = question

    common = {
        "a", "an", "and", "are", "as", "bhutan", "can", "current", "do", "does", "for", "how", "i", "in",
        "is", "latest", "me", "my", "now", "of", "on", "please", "rate", "tax", "taxes", "tell", "the", "to",
        "today", "tomorrow", "was", "what", "when", "where", "which", "will",
    }
    focus = {term for term in bm25_tokenize(phrase) if term not in common and not term.isdigit()}
    return bool(focus) and any(focus <= set(bm25_tokenize(item.get("text", ""))) for item in evidence)


def expand_retrieval_query(query: str) -> str:
    """Add statutory vocabulary for common plain-language GST questions."""
    expansions: list[str] = []
    if gst_rate_intent(query):
        expansions.append("Schedule I applicable GST rate taxable supplies taxable imports value of supply import")
    if vehicle_context(query):
        expansions.append("passenger vehicle motor car purchase import")
    return " ".join([query, *expansions]).strip()


def rate_passage_adjustment(query: str, item: dict) -> float:
    if not gst_rate_intent(query):
        return 0.0
    text = str(item.get("text", "")).lower()
    if "gst rate" in text and re.search(r"\d+(?:\.\d+)?\s*%", text):
        return 0.16
    if "schedule i" in text and "gst rate" in text:
        return 0.045
    return 0.0


def toc_like(text: str) -> bool:
    """Identify contents/index fragments that should not outrank operative law."""
    compact = re.sub(r"\s+", " ", text).strip()
    dotted_leader = bool(re.search(r"\.{6,}\s*\d*", compact))
    contents_page = "contents" in compact.lower() and compact.count("...") >= 2
    repeated_leaders = compact.count("...") >= 2
    return dotted_leader or contents_page or repeated_leaders


def substantive_legal_text(text: str) -> bool:
    """Return True for readable evidence rather than headings or contents rows."""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    return (
        len(compact.split()) >= 18
        and not toc_like(compact)
        and not bool(re.fullmatch(r"[\W\d_]+", compact))
    )


def substantive_adjustment(item: dict) -> float:
    """Small reranking adjustment for readable, operative legal passages."""
    text = str(item.get("text", ""))
    if toc_like(text):
        return -0.12
    adjustment = 0.0
    if len(text) >= 140:
        adjustment += 0.012
    if re.search(r"\b(may|must|shall|is amended as|is repealed|is inserted)\b", text, re.I):
        adjustment += 0.012
    return adjustment


def best_answer_evidence(evidence: list[dict]) -> dict:
    """Prefer a substantive passage over a contents/index fragment."""
    return next(
        (
            item
            for item in evidence
            if not toc_like(str(item.get("text", ""))) and len(str(item.get("text", ""))) >= 60
        ),
        evidence[0],
    )


def weak_generated_answer(answer: str) -> bool:
    """Reject labels and fragments that cannot communicate a legal answer."""
    cleaned = re.sub(r"\s+", " ", answer).strip(" .:-")
    tokens = bm25_tokenize(cleaned)
    label_only = cleaned.lower() in {
        "change",
        "changed",
        "amendment",
        "amended",
        "yes",
        "no",
        "applicable",
        "not applicable",
    }
    return label_only or len(tokens) < 8


def evidence_supports_question(question: str, evidence: dict) -> bool:
    """Conservative check for using an extractive fallback after LLM failure."""
    text = str(evidence.get("text", ""))
    if toc_like(text) or len(text) < 60:
        return False
    section_match = re.search(r"\bsection\s+(\d{1,4}[a-z]?(?:\s*\([^)]+\))?)", question, re.I)
    if section_match:
        requested = re.sub(r"\s+", "", section_match.group(1)).lower()
        stated = re.sub(r"\s+", "", str(evidence.get("section_number", ""))).lower()
        affected = [re.sub(r"\s+", "", str(value)).lower() for value in evidence.get("affected_sections", [])]
        mentioned = bool(re.search(rf"\bsection\s+{re.escape(section_match.group(1))}\b", text, re.I))
        return requested == stated or requested in affected or mentioned
    query_terms = set(bm25_tokenize(question)) - {
        "what", "which", "does", "the", "and", "for", "from", "under", "about", "section"
    }
    return len(query_terms & set(bm25_tokenize(text))) >= 3
