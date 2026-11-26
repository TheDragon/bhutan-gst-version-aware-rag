"""Amendment history of one provision, for the Version timeline page.

The page used to list every passage whose section number matched (schedule
rows, other documents, unrelated clauses that happen to be numbered 40). This
module instead:

1. finds the provision's own text (e.g. GST Act Section 40 is the passage in
   the Act that *starts* "40. ", not a schedule row on page 316);
2. builds its version chain with ``VersionIndex`` (original Act -> later
   versions);
3. attaches the amending clauses recorded in ``amendment_events.csv`` to the
   version they created, with a plain description of the change.

Everything else that merely shares the number is returned separately as
``other_matches``.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .version_view import SHORT_NAMES, VersionIndex, _diff, _focus, _split_notes, _words

INSTRUMENTS = {
    "gst_act_2020": {"label": "GST Act", "locator": "Section", "documents": ["gst_act_2020", "gst_consolidated_2025"]},
    "gst_rules_2026": {"label": "GST Rules", "locator": "Rule", "documents": ["gst_rules_2026"]},
    "gst_amendment_2021": {"label": "2021 Amendment Act", "locator": "Section", "documents": ["gst_amendment_2021"]},
}
SCHEDULE_LABELS = {
    "scheduleivc": "Schedule IV-C (exemption list)",
    "scheduleix": "Schedule IX",
    "schedulex": "Schedule X (transitional provisions)",
}
NOW_READS = re.compile(
    r"\b(?:is|are) (?:amended|substituted|replaced) as\s*:?\s*|\binserted,?\s*namely\s*:?\s*|\bis amended\.\s+",
    re.I,
)


def _parent(section: str) -> str:
    """'264(4)' -> '264'; '231a' -> '231a'; 'scheduleivc' stays as is."""
    match = re.match(r"(\d+[a-z]?)", section.lower())
    return match.group(1) if match else section.lower()


def _norm(section: str) -> str:
    return re.sub(r"[\s.]+", "", str(section)).lower()


def _plain(text: str) -> str:
    return " ".join(str(text).split())


def _first_words(text: str, limit: int = 12) -> str:
    text = re.sub(r"^\d{1,4}[A-Za-z]?\.\s+", "", _plain(text))
    sentence = re.split(r"(?<=[.;:])\s", text, maxsplit=1)[0]
    words = sentence.split()
    return " ".join(words[:limit]) + ("…" if len(words) > limit else "")


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def describe_change(ops: list[dict], limit: int = 3) -> str:
    """Turn diff operations into 'Removed “or EET”; replaced “7%” with “5%”'."""
    phrases: list[str] = []
    pending_del = ""
    for op in ops + [{"op": "same", "text": ""}]:
        if op["op"] == "del":
            pending_del = f"{pending_del} {op['text']}".strip()
        elif op["op"] == "add":
            added = op["text"]
            if _alnum(added) == _alnum(pending_del):
                pass  # punctuation or spacing only
            elif not _alnum(pending_del):
                if _alnum(added):
                    phrases.append(f"added “{_clip(added)}”")
            elif not _alnum(added):
                phrases.append(f"removed “{_clip(pending_del)}”")
            else:
                phrases.append(f"replaced “{_clip(pending_del)}” with “{_clip(added)}”")
            pending_del = ""
        else:
            if _alnum(pending_del):
                phrases.append(f"removed “{_clip(pending_del)}”")
            pending_del = ""
    if not phrases:
        return ""
    shown = "; ".join(phrases[:limit])
    more = len(phrases) - limit
    text = shown + (f"; and {more} more change{'s' if more > 1 else ''}" if more > 0 else "")
    return text[0].upper() + text[1:]


def _clip(text: str, size: int = 70) -> str:
    text = text.strip(" ,;")
    return text if len(text) <= size else text[: size - 1].rstrip() + "…"


@dataclass
class Event:
    event_id: str
    source_document_id: str
    affected_section: str
    normalized: str
    relation: str
    source_chunk_id: str
    effective_from: str
    target_instrument: str
    target_scope: str
    excerpt: str


class ProvisionStories:
    def __init__(self, chunks: list[dict], events_csv: Path, versions: VersionIndex | None = None):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.versions = versions or VersionIndex(chunks)
        with open(events_csv, encoding="utf-8") as handle:
            self.events = [
                Event(
                    event_id=row["event_id"],
                    source_document_id=row["source_document_id"],
                    affected_section=row["affected_section"],
                    normalized=_norm(row["affected_section_normalized"] or row["affected_section"]),
                    relation=row["relation_type"],
                    source_chunk_id=row["source_chunk_id"],
                    effective_from=row["effective_from"],
                    target_instrument=row["target_instrument"],
                    target_scope=row["target_scope"],
                    excerpt=_plain(row["evidence_excerpt"]),
                )
                for row in csv.DictReader(handle)
            ]

    # -- locating the provision ------------------------------------------------
    def _body(self, document_id: str) -> dict[str, dict]:
        """Section number -> passage for the document's numbered body.

        Walks the document in page order and only accepts a passage starting
        '<n>. ' when n continues the numbering (a schedule restarting at '1.'
        or a paragraph '40.' on page 340 is not Section 40).
        """
        cache = self.__dict__.setdefault("_body_cache", {})
        if document_id in cache:
            return cache[document_id]
        body: dict[str, dict] = {}
        last = 0
        rows = sorted((c for c in self.chunks if c["document_id"] == document_id), key=lambda c: (c.get("page_number") or 0))
        for chunk in rows:
            match = re.match(r"^(\d{1,4})([A-Za-z]?)\s*\.\s", _plain(chunk.get("text", "")))
            if not match or _norm(chunk.get("section_number", "")) != _norm(match.group(1) + match.group(2)):
                continue
            number = int(match.group(1))
            if last < number <= last + 25 or (number == last and match.group(2)):
                body.setdefault(_norm(match.group(1) + match.group(2)), chunk)
                last = number
        cache[document_id] = body
        return body

    def main_chunk(self, instrument: str, section: str) -> dict | None:
        """The passage that *is* the provision, from the first document that has it."""
        parent = _parent(section)
        if not re.fullmatch(r"\d+[a-z]?", parent):
            return None
        for document_id in INSTRUMENTS.get(instrument, {}).get("documents", []):
            chunk = self._body(document_id).get(parent)
            if chunk:
                return chunk
        return None

    def events_for(self, instrument: str, section: str) -> list[Event]:
        key = _norm(section)
        if key in SCHEDULE_LABELS:
            return [e for e in self.events if e.target_instrument == instrument and _norm(e.target_scope) == key]
        parent = _parent(key)
        return [
            e for e in self.events
            if e.target_instrument == instrument and not _norm(e.target_scope).startswith("schedule")
            and (e.normalized == key or _parent(e.normalized) == parent)
        ]

    def title(self, instrument: str, section: str) -> str:
        chunk = self.main_chunk(instrument, section)
        if chunk:
            return _first_words(chunk.get("text", ""))
        if _norm(section) in SCHEDULE_LABELS:
            return SCHEDULE_LABELS[_norm(section)]
        events = self.events_for(instrument, section)
        return _first_words(events[0].excerpt) if events else ""

    # -- the story -------------------------------------------------------------
    def story(self, instrument: str, section: str, today: str | None = None) -> dict:
        today = today or date.today().isoformat()
        info = INSTRUMENTS.get(instrument, INSTRUMENTS["gst_act_2020"])
        key = _norm(section)
        events = sorted(self.events_for(instrument, key), key=lambda e: (e.effective_from, e.event_id))
        base = self.main_chunk(instrument, key)
        label = SCHEDULE_LABELS.get(key) or f"{info['locator']} {section.strip()}"

        nodes: list[dict] = []
        chain_ids: set[str] = set()
        # When the original Act's own text for this section wasn't extracted,
        # the story starts from the consolidated text instead.
        base_is_consolidated = bool(base) and base.get("document_type") == "consolidation"
        base_start = (base or {}).get("version_start") or (base or {}).get("effective_from") or ""
        if base:
            ctx = self.versions.context(base, today) or {}
            timeline = ctx.get("timeline") or []
            previous_item = None
            for row in timeline:
                item = self.by_id[row["chunk_id"]]
                chain_ids.add(item["chunk_id"])
                node = {
                    "start": row["start"],
                    "end": row["end"],
                    "short_name": row["short_name"],
                    "title": row["title"],
                    "page": row["page_number"],
                    "source_url": row["source_url"],
                    "chunk_id": item["chunk_id"],
                    "text": _split_notes(_plain(re.sub(r"^\s*\d{1,4}[A-Za-z]?\.\s+", "", item["text"])))[0],
                    "kind": "original" if previous_item is None else "amended",
                    "diff": [],
                    "change": "",
                    "clauses": [],
                    "also_in": [],
                }
                if previous_item is not None:
                    shared = _words(previous_item["text"]) & _words(item["text"])
                    old_text, _ = _split_notes(_focus(previous_item["text"], shared))
                    new_text, _ = _split_notes(_focus(item["text"], shared))
                    ops, ratio = _diff(old_text, new_text)
                    changed = any(op["op"] != "same" for op in ops)
                    has_clause = any(e.effective_from == row["start"] for e in events)
                    if not changed and not has_clause:
                        # Same wording reprinted (e.g. in the consolidated Act): not a new version.
                        nodes[-1]["also_in"].append({"short_name": row["short_name"], "page": row["page_number"], "source_url": row["source_url"], "start": row["start"]})
                        nodes[-1]["end"] = row["end"]
                        nodes[-1]["current_chunk_id"] = item["chunk_id"]
                        previous_item = item
                        continue
                    node["diff"] = ops if ratio >= 0.35 else []
                    node["change"] = describe_change(ops) if ratio >= 0.35 else "Rewritten; compare the full texts."
                    node["before_text"], node["after_text"] = old_text, new_text
                nodes.append(node)
                previous_item = item

        # Attach the amending clauses to the version they created.
        for event in events:
            clause_chunk = self.by_id.get(event.source_chunk_id, {})
            chain_ids.add(event.source_chunk_id)
            clause = {
                "event_id": event.event_id,
                "relation": event.relation,
                "affected": event.affected_section,
                "short_name": SHORT_NAMES.get(event.source_document_id, event.source_document_id),
                "clause": _clause_number(event.excerpt) or str(clause_chunk.get("section_number", "")),
                "page": clause_chunk.get("page_number"),
                "source_url": clause_chunk.get("source_url", ""),
                "text": re.sub(r"^(?:\d{4}\.\s+)?\d{1,3}\.\s+", "", event.excerpt)[:900],
                "now_reads": _now_reads(event),
                "effective_from": event.effective_from,
            }
            target = next(
                (n for n in nodes if n["start"] == event.effective_from and (n["kind"] != "original" or base_is_consolidated)),
                None,
            )
            if target is None and event.relation == "repealed" and _parent(event.normalized) == key:
                target = {
                    "start": event.effective_from, "end": "", "short_name": clause["short_name"], "title": "",
                    "page": clause["page"], "source_url": clause["source_url"], "chunk_id": event.source_chunk_id,
                    "text": "", "kind": "repealed", "diff": [], "change": "Repealed: this provision no longer applies.",
                    "clauses": [], "also_in": [],
                }
                if nodes and not nodes[-1]["end"]:
                    nodes[-1]["end"] = _day_before(event.effective_from)
                nodes.append(target)
            if target is None:
                # Repealing or inserting a *part* (264(6), a Schedule X paragraph)
                # amends the provision; only repealing the whole thing repeals it.
                kind = "inserted" if event.relation == "inserted" else "amended"
                target = {
                    "start": event.effective_from, "end": "", "short_name": clause["short_name"], "title": "",
                    "page": clause["page"], "source_url": clause["source_url"], "chunk_id": event.source_chunk_id,
                    "text": "", "kind": kind, "diff": [],
                    "change": _relation_text(event), "clauses": [], "also_in": [],
                }
                nodes.append(target)
            target["clauses"].append(clause)

        notices = self._notices_about(key)
        for node in nodes:
            node["notices"] = [n for n in notices if n["start"] >= node["start"]] if node["kind"] != "original" else []
            if base_is_consolidated and node["kind"] == "original" and node["clauses"]:
                node["kind"] = "amended"
            clauses = node["clauses"]
            if len(clauses) >= 3:
                # e.g. 15 definitions in Section 337 changed at once: a passage
                # diff is noise; summarise and let each clause speak.
                counts = {r: sum(1 for c in clauses if c["relation"] == r) for r in ("amended", "inserted", "repealed")}
                parts = [f"{n} {r}" for r, n in counts.items() if n]
                node["change"] = f"{len(clauses)} parts changed ({', '.join(parts)}); see the amending clauses."
                node["diff"] = []
            elif clauses and (not node["change"] or not node["diff"]):
                node["change"] = " ".join(c["now_reads"] for c in clauses if c["now_reads"]) or node["change"]
            if node["notices"] and node["change"].endswith("was amended."):
                # The amending clause only names the schedule; the official
                # notice says what was actually added.
                notice = re.sub(r"^Background\s+", "", node["notices"][0]["text"])
                node["change"] = f"{node['change']} The official notice explains: “{_clip(notice, 260)}”"

        nodes.sort(key=lambda n: (n["start"], {"original": 0, "amended": 1, "inserted": 1, "repealed": 2}.get(n["kind"], 1)))
        repealed = any(n["kind"] == "repealed" for n in nodes)
        current = None
        latest = nodes[-1] if nodes else None
        if not repealed and latest is not None and (latest["text"] or latest["kind"] == "original"):
            current = next((n for n in reversed(nodes) if n["text"] and n["start"] <= today), None)
        changes = [n for n in nodes if n["kind"] != "original"]
        status = "repealed" if repealed else ("amended" if changes else ("unchanged" if nodes else "not_found"))

        return {
            "instrument": instrument,
            "instrument_label": info["label"],
            "section": section.strip(),
            "label": label,
            "title": self.title(instrument, key),
            "status": status,
            "summary": _summary(status, changes, nodes),
            "current": _current_view(current, nodes) if current else None,
            # Only when nothing earlier than the consolidated text was found.
            "original_missing": base_is_consolidated and not any(n["start"] < base_start for n in nodes),
            "history": list(reversed(nodes)),
            "other_matches": self._other_matches(key, chain_ids),
        }

    def _notices_about(self, key: str) -> list[dict]:
        """Official notices that explain a change to a schedule (e.g. what was added to Schedule IV-C)."""
        if key not in SCHEDULE_LABELS:
            return []
        letters = key.removeprefix("schedule")
        mention = re.compile(rf"Schedule\s+{letters[:-1]}\s*\(?\s*{letters[-1]}\b" if len(letters) > 2 else rf"Schedule\s+{letters}\b", re.I)
        rows = []
        for c in self.chunks:
            text = _plain(c.get("text", ""))
            if c.get("document_type") == "notice" and mention.search(text) and re.search(r"\b(added|exempt|takes? effect)\b", text, re.I):
                rows.append({
                    "short_name": SHORT_NAMES.get(c["document_id"], c.get("title", "")),
                    "page": c.get("page_number"),
                    "start": c.get("version_start") or c.get("effective_from") or "",
                    "text": re.sub(r"^\d{1,2}\.\s+", "", text)[:420],
                    "source_url": c.get("source_url", ""),
                })
        return rows[:2]

    def _other_matches(self, section: str, exclude: set[str]) -> list[dict]:
        parent = _parent(section)
        rows = []
        for c in self.chunks:
            if c["chunk_id"] in exclude or _norm(c.get("section_number", "")) != parent:
                continue
            rows.append({
                "chunk_id": c["chunk_id"],
                "short_name": SHORT_NAMES.get(c["document_id"], c.get("title", "")),
                "document_type": c.get("document_type", ""),
                "page": c.get("page_number"),
                "preview": _plain(c.get("text", ""))[:140],
                "source_url": c.get("source_url", ""),
            })
        return sorted(rows, key=lambda r: (r["short_name"], r["page"] or 0))

    # -- the index of everything that changed ----------------------------------
    def index(self) -> list[dict]:
        groups: dict[str, dict] = {}
        seen: set[tuple[str, str, str]] = set()
        for event in sorted(self.events, key=lambda e: (e.effective_from, e.source_document_id)):
            group = groups.setdefault(event.source_document_id, {
                "document_id": event.source_document_id,
                "short_name": SHORT_NAMES.get(event.source_document_id, event.source_document_id),
                "effective_from": event.effective_from,
                "items": [],
            })
            scope = _norm(event.target_scope)
            # One entry per provision (or per schedule); its story lists every clause.
            parent = scope if scope in SCHEDULE_LABELS else _parent(event.normalized)
            key = (event.source_document_id, event.target_instrument, parent)
            whole_repeal = event.relation == "repealed" and event.normalized == parent
            if key in seen:
                if whole_repeal:
                    next(i for i in group["items"] if i["_key"] == key)["relation"] = "repealed"
                continue
            seen.add(key)
            group["items"].append({
                "_key": key,
                "instrument": event.target_instrument,
                "section": parent if parent in SCHEDULE_LABELS else event.affected_section.split("(")[0].strip(),
                "label": _item_label(event, parent),
                # A part repealed or inserted (264(6), 231A) still means the provision was amended.
                "relation": "repealed" if whole_repeal else "amended",
                "title": self.title(event.target_instrument, parent) or _first_words(event.excerpt),
            })
        for group in groups.values():
            for item in group["items"]:
                item.pop("_key", None)
        for group in groups.values():
            group["counts"] = {r: sum(1 for i in group["items"] if i["relation"] == r) for r in ("amended", "inserted", "repealed")}
        return sorted(groups.values(), key=lambda g: g["effective_from"], reverse=True)


def _item_label(event: Event, parent: str) -> str:
    if parent in SCHEDULE_LABELS:
        return SCHEDULE_LABELS[parent]
    if event.target_instrument == "gst_amendment_2021":
        return f"2021 Amendment, section {parent}"
    if event.target_scope.startswith("Schedule"):
        return f"{event.target_scope}, section {event.affected_section}"
    return f"Section {event.affected_section.split('(')[0].strip()}"


def _clause_number(excerpt: str) -> str:
    """'2022. 3. In the …' -> '3'; '12. Section 40 …' -> '12'."""
    for number in re.findall(r"(?<!\S)(\d{1,4})\.(?=\s)", excerpt[:40]):
        if len(number) <= 3:
            return number
    return ""


def _now_reads(event: Event) -> str:
    """Plain description of what an amending clause does."""
    target = event.affected_section
    if _norm(event.target_scope).startswith("schedule"):
        name = f"{event.target_scope} paragraph {target}" if re.match(r"\d", target) else target
    else:
        name = f"Section {target}" if re.match(r"\d", target) else target
    if event.relation == "repealed":
        return f"{name} was repealed."
    match = NOW_READS.search(event.excerpt)
    if not match:
        return f"{name} was amended." if event.relation == "amended" else f"New text inserted after {name}."
    new_text = event.excerpt[match.end():].strip(" “”\"")
    new_text = re.sub(r"\s+\d{1,3}$", "", new_text)  # trailing page number
    if not new_text:
        return f"{name} was amended."
    if event.relation == "inserted":
        lead = "New provision inserted"
    elif re.search(r"\bTitle of\b", event.excerpt[:60], re.I):
        lead = f"The title of {name} now reads"
    else:
        lead = f"{name} now reads"
    return f"{lead}: “{_clip(new_text, 220)}”"


def _relation_text(event: Event) -> str:
    return {
        "inserted": f"A new provision was inserted after {event.affected_section}.",
        "repealed": f"{event.affected_section} was repealed.",
    }.get(event.relation, f"{event.affected_section} was amended.")


def _day_before(iso: str) -> str:
    from datetime import timedelta

    return (date.fromisoformat(iso) - timedelta(days=1)).isoformat()


def readable_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{parsed.day} {parsed:%B %Y}"


def _summary(status: str, changes: list[dict], nodes: list[dict]) -> str:
    if status == "not_found":
        return "No text or recorded amendment was found for this provision."
    if status == "unchanged":
        return f"Unchanged since {readable_date(nodes[0]['start'])}." if nodes else "Unchanged."
    last = changes[-1]
    clause = last["clauses"][0] if last["clauses"] else None
    by = f" by the {clause['short_name']}" + (f" (clause {clause['clause']})" if clause and clause["clause"] else "") if clause else ""
    if status == "repealed":
        return f"Repealed on {readable_date(last['start'])}{by}."
    count = len(changes)
    times = "once" if count == 1 else f"{count} times"
    return f"Amended {times} · last changed on {readable_date(last['start'])}{by}."


def _current_view(current: dict, nodes: list[dict]) -> dict:
    latest_change = next((n for n in reversed(nodes) if n["kind"] != "original" and n["diff"]), None)
    return {
        "start": current["start"],
        "short_name": (current["also_in"][-1]["short_name"] if current.get("also_in") else current["short_name"]),
        "page": (current["also_in"][-1]["page"] if current.get("also_in") else current["page"]),
        "source_url": (current["also_in"][-1]["source_url"] if current.get("also_in") else current["source_url"]),
        "text": current["text"],
        "diff": latest_change["diff"] if latest_change is current else [],
    }
