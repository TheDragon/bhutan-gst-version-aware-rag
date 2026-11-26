"""Version history for the passage an answer relies on.

Given the passage an answer cites, this module finds the other versions of the
same provision across the corpus (original Act, amendments, consolidated Act,
rules, notices) and reports, for the user's legal date:

* a timeline of versions, each marked in force / superseded / not yet in force;
* the version before (or after) the cited one, with a word-level diff;
* the versions that were set aside for that date and why.

Matching uses the corpus metadata (``version_family``, ``affected_sections``)
plus word overlap, because amendment clauses and schedule rows do not always
share a family id with the provision they change.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import date, timedelta

from .quality import bm25_tokenize, toc_like

SHORT_NAMES = {
    "gst_act_2020": "GST Act 2020",
    "gst_amendment_2021": "2021 Amendment",
    "gst_amendment_2022": "2022 Amendment",
    "gst_amendment_2025": "2025 Amendment",
    "gst_amendment_2026": "2026 Amendment",
    "gst_consolidated_2025": "Consolidated Act",
    "gst_rules_2026": "GST Rules 2026",
    "gst_rule_change_notice_2026": "Rule change notice",
    "gst_implementation_notice_2025": "GST launch notice",
    "gst_implementation_notice_2026": "2026 exemption notice",
}
# When two documents carry the same version on the same date, show the one a
# reader can use directly (the consolidated text) rather than the amending clause.
DOC_PREFERENCE = {"consolidation": 0, "act": 1, "rules": 2, "amendment": 3, "notice": 4}
STOP = {
    "the", "and", "for", "shall", "with", "that", "this", "from", "under", "any", "such", "which", "may",
    "are", "was", "were", "been", "has", "have", "not", "other", "than", "into", "its", "their", "per",
    "act", "section", "bhutan", "amended", "amendment", "chapter", "part", "goods", "services", "tax",
}
CONTAINED = 0.7       # share of the cited passage's words found in another passage
RELATED = 0.3         # lower bar when metadata already links the passages
MAX_TEXT = 900


def _words(text: str) -> set[str]:
    return {t for t in bm25_tokenize(text) if len(t) > 2 and t not in STOP and not t.isdigit()}


def _norm_section(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _day_before(iso: str) -> str:
    return (date.fromisoformat(iso) - timedelta(days=1)).isoformat()


def _start(item: dict) -> str:
    return item.get("version_start") or item.get("effective_from") or ""


def _clean(text: str) -> str:
    text = " ".join(str(text).split())
    text = re.sub(r"^\d{1,4}[A-Za-z]?\.\s+", "", text)
    # Amending clauses wrap the new wording: 'Section 40 of ... is amended as: "…"'.
    text = re.sub(r"^.{0,260}?\bis (?:amended|substituted|replaced) as:\s*", "", text, flags=re.I)
    return text.strip(" “”\"")


def _focus(text: str, anchor_words: set[str]) -> str:
    """Keep the sentences of ``text`` that talk about the same thing as the anchor."""
    sentences = re.split(r"(?<=[.;:])\s+", _clean(text))
    kept = [
        s for s in sentences
        if len(s.split()) >= 3
        # Headings of the next schedule/chapter often share a chunk with the clause.
        and not re.match(r"(?:SCHEDULE|CHAPTER|PART)\b", s)
        and (w := _words(s)) and len(w & anchor_words) / len(w) >= 0.5
    ]
    focused = " ".join(kept) if kept else _clean(text)
    return focused[:MAX_TEXT]


AMENDED_NOTE = re.compile(r"\s*\[(?:Amended|Inserted|Substituted|Repealed) by ([^\]]+)\]", re.I)
KEEP_CONTEXT = 22  # unchanged words kept on each side of a change


def _split_notes(text: str) -> tuple[str, list[str]]:
    """Remove the consolidated Act's '[Amended by …]' notes and return them."""
    notes = [" ".join(n.split()) for n in AMENDED_NOTE.findall(text)]
    return " ".join(AMENDED_NOTE.sub("", text).split()), notes


def _key(word: str) -> str:
    return re.sub(r"[-‐–]", "", word.lower())


def _diff(before: str, after: str) -> tuple[list[dict], float]:
    # Compare words without case or hyphens so "zero rate" vs "zero-rate" is not a change.
    a = re.sub(r"(\w)[-‐–](\w)", r"\1 \2", before).split()
    b = re.sub(r"(\w)[-‐–](\w)", r"\1 \2", after).split()
    a_keys = [_key(w) for w in a]
    b_keys = [_key(w) for w in b]
    matcher = difflib.SequenceMatcher(a=a_keys, b=b_keys, autojunk=False)
    ops: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ops.append({"op": "same", "text": b[j1:j2]})
            continue
        if tag in {"replace", "delete"}:
            ops.append({"op": "del", "text": a[i1:i2]})
        if tag in {"replace", "insert"}:
            ops.append({"op": "add", "text": b[j1:j2]})
    # Trim long unchanged stretches so the change itself stands out.
    for index, op in enumerate(ops):
        words = op.pop("text")
        if op["op"] == "same" and len(words) > 2 * KEEP_CONTEXT:
            first, last = index == 0, index == len(ops) - 1
            if first:
                words = ["…", *words[-KEEP_CONTEXT:]]
            elif last:
                words = [*words[:KEEP_CONTEXT], "…"]
            else:
                words = [*words[:KEEP_CONTEXT], "…", *words[-KEEP_CONTEXT:]]
        op["text"] = " ".join(words)
    return ops, matcher.ratio()


@dataclass
class _Entry:
    item: dict
    start: str
    end: str = ""


class VersionIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = [c for c in chunks if not toc_like(str(c.get("text", "")))]
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.words = {c["chunk_id"]: _words(c.get("text", "")) for c in self.chunks}

    # -- finding the versions -------------------------------------------------
    def _related(self, cited: dict) -> list[dict]:
        anchor = self.words.get(cited["chunk_id"]) or _words(cited.get("text", ""))
        if len(anchor) < 6:
            return [cited]
        family = cited.get("version_family", "")
        linked_family = family if family and not family.startswith("amendment:") else ""
        section = _norm_section(cited.get("section_number"))
        best: dict[str, tuple[float, dict]] = {}
        cited_is_notice = cited.get("document_type") == "notice"
        for item in self.chunks:
            if item["chunk_id"] == cited["chunk_id"] or item["document_id"] == cited["document_id"]:
                continue
            # A notice implements or explains the law; it is not another
            # version of a provision. Linking them produces false "changes"
            # (e.g. the same schedule split at different page breaks).
            if cited_is_notice or item.get("document_type") == "notice":
                continue
            overlap = len(anchor & self.words[item["chunk_id"]]) / len(anchor)
            linked = (linked_family and item.get("version_family") == linked_family) or (
                section and section in {_norm_section(s) for s in item.get("affected_sections", [])}
            )
            if overlap >= CONTAINED or (linked and overlap >= RELATED):
                if overlap > best.get(item["document_id"], (-1.0, None))[0]:
                    best[item["document_id"]] = (overlap, item)
        return [cited, *(item for _, item in best.values())]

    def _entries(self, cited: dict) -> list[_Entry]:
        by_start: dict[str, dict] = {}
        for item in self._related(cited):
            start = _start(item)
            if not start:
                continue
            current = by_start.get(start)
            if current is None or item is cited or (
                current is not cited
                and DOC_PREFERENCE.get(item.get("document_type"), 9) < DOC_PREFERENCE.get(current.get("document_type"), 9)
            ):
                by_start[start] = item
        entries = [_Entry(item, start) for start, item in sorted(by_start.items())]
        for i, entry in enumerate(entries):
            recorded = entry.item.get("version_end") or entry.item.get("effective_to") or ""
            implied = _day_before(entries[i + 1].start) if i + 1 < len(entries) else ""
            # Metadata sometimes leaves an older version open-ended; a later
            # version of the same provision closes it.
            entry.end = min(filter(None, (recorded, implied)), default="")
        return entries

    # -- public ----------------------------------------------------------------
    def context(self, cited: dict | None, target_date: str | None) -> dict | None:
        if not cited:
            return None
        entries = self._entries(cited)
        target = target_date or date.today().isoformat()

        def status(entry: _Entry) -> str:
            if entry.start > target:
                return "not_yet"
            if entry.end and entry.end < target:
                return "superseded"
            return "in_force"

        def describe(entry: _Entry) -> dict:
            item = entry.item
            return {
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "short_name": SHORT_NAMES.get(item["document_id"], item.get("title", "")),
                "title": item.get("title", ""),
                "document_type": item.get("document_type", ""),
                "affected_sections": list(item.get("affected_sections", [])),
                "section_number": str(item.get("section_number", "")),
                "page_number": item.get("page_number"),
                "start": entry.start,
                "end": entry.end,
                "status": status(entry),
                "source_url": item.get("source_url", ""),
                "selected": item is cited,
            }

        timeline = [describe(e) for e in entries]
        position = next((i for i, e in enumerate(entries) if e.item is cited), None)
        if position is None:  # cited passage has no start date
            return {"selected": None, "timeline": [], "compare": None, "set_aside": []}

        previous = entries[position - 1] if position > 0 else None
        following = entries[position + 1] if position + 1 < len(entries) else None
        compare = None
        pair = (previous, entries[position], "earlier") if previous else (
            (entries[position], following, "later") if following else None
        )
        if pair:
            old, new, direction = pair
            # Focus both texts on what they share, so a chunk that also holds
            # the next schedule or section doesn't swamp the comparison.
            shared = (self.words.get(old.item["chunk_id"]) or set()) & (self.words.get(new.item["chunk_id"]) or set())
            old_raw, old_notes = _split_notes(old.item["text"])
            new_raw, new_notes = _split_notes(new.item["text"])
            old_text, new_text = _focus(old_raw, shared), _focus(new_raw, shared)
            ops, ratio = _diff(old_text, new_text)
            if any(op["op"] != "same" for op in ops):
                compare = {
                    "direction": direction,
                    "before": describe(old) | {"text": old_text},
                    "after": describe(new) | {"text": new_text},
                    "diff": ops if ratio >= 0.35 else [],
                    "similarity": round(ratio, 3),
                    "amended_by": next(iter(new_notes or old_notes), ""),
                }

        set_aside = [
            {**row, "reason": _reason(row)}
            for row in timeline
            if not row["selected"] and row["status"] != "in_force"
        ]
        return {
            "selected": timeline[position],
            "timeline": timeline,
            "compare": compare,
            "set_aside": set_aside,
        }

    def related_passages(self, cited: dict | None, target_date: str | None) -> list[dict]:
        """Earlier/later versions to show the answer model, labelled for the prompt."""
        ctx = self.context(cited, target_date)
        if not ctx or not ctx["compare"]:
            return []
        other = ctx["compare"]["before" if ctx["compare"]["direction"] == "earlier" else "after"]
        item = self.by_id.get(other["chunk_id"])
        return [item | {"_version_label": "EARLIER VERSION" if ctx["compare"]["direction"] == "earlier" else "LATER VERSION"}] if item else []


def compare_covers_quote(ctx: dict | None, quote: str, threshold: float = 0.6) -> bool:
    """True when the before/after comparison is about the wording that was quoted.

    Long passages (a definitions section, a schedule) can change in one place
    while the answer quotes another; the amendment shown must be the relevant one.
    """
    compare = (ctx or {}).get("compare")
    if not compare:
        return False
    def flat(text: str) -> str:
        return " " + " ".join(bm25_tokenize(text)) + " "

    quoted = flat(quote)
    if not quoted.strip():
        return True
    before, after = flat(compare["before"].get("text", "")), flat(compare["after"].get("text", ""))
    if quoted in before and quoted in after:
        return False  # the quoted wording is identical in both versions; the change is elsewhere
    if quoted in before or quoted in after:
        return True   # the quoted wording exists in one version only, so it is what changed

    # The quote was trimmed out of the comparison text: fall back to word overlap.
    def tokens(text: str) -> set[str]:
        return {t for t in bm25_tokenize(text) if len(t) > 2 and not t.isdigit()}

    quote_words = tokens(quote)
    used = compare["before"] if compare["before"].get("selected") else compare["after"]
    return len(quote_words & tokens(used.get("text", ""))) / len(quote_words) >= threshold


def _reason(row: dict) -> str:
    if row["status"] == "superseded":
        return f"Replaced after {row['end']}" if row["end"] else "Replaced by a later version"
    if row["status"] == "not_yet":
        return f"Not in force until {row['start']}"
    return ""


def follow_up_dates(ctx: dict | None) -> list[dict]:
    """One-click re-asks of the same question at another version's date."""
    if not ctx or not ctx.get("selected"):
        return []
    suggestions = []
    timeline = ctx["timeline"]
    position = next(i for i, row in enumerate(timeline) if row["selected"])
    if position > 0:
        earlier = timeline[position - 1]
        when = earlier["end"] or earlier["start"]
        suggestions.append({"kind": "date", "target_date": when, "label": f"Ask about {earlier['short_name']} ({when})"})
    if position + 1 < len(timeline):
        later = timeline[position + 1]
        suggestions.append({"kind": "date", "target_date": later["start"], "label": f"Ask about {later['short_name']} ({later['start']})"})
    return suggestions
