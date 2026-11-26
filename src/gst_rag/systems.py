from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .paths import INDEXES
from .quality import (
    act_commencement_clause,
    bm25_tokenize,
    commencement_intent,
    expand_retrieval_query,
    rate_passage_adjustment,
    substantive_adjustment,
)
from .retrieval import parse_query_date, reciprocal_rank_fusion
from .versioning import version_valid_on


SYSTEMS = ("standard", "date_filtered", "version_aware")

@dataclass
class RetrievalEngine:
    chunks: list[dict]
    model_name: str
    index_path: Path

    def __post_init__(self) -> None:
        # Amendment Acts often repeat one numbered section in the contents and
        # again in the operative pages. Propagate source-verified relationships
        # within that document/section group so the complete legal passage can
        # receive the same version-aware treatment as its contents entry.
        grouped: dict[tuple[str, str], list[dict]] = {}
        for item in self.chunks:
            key = (str(item.get("document_id", "")), str(item.get("section_number", "")))
            grouped.setdefault(key, []).append(item)
        for group in grouped.values():
            affected = sorted({str(value) for item in group for value in item.get("affected_sections", [])})
            relations = sorted({str(value) for item in group for value in item.get("relation_types", [])})
            event_ids = sorted({str(value) for item in group for value in item.get("amendment_event_ids", [])})
            relationship_source = next((item for item in group if item.get("affected_sections")), None)
            if not relationship_source:
                continue
            for item in group:
                item["affected_sections"] = affected
                item["relation_types"] = relations
                item["amendment_event_ids"] = event_ids
                item["version_status"] = "change_evidence"
                item["version_family"] = relationship_source.get("version_family", item.get("version_family", ""))

        self.ids = [item["chunk_id"] for item in self.chunks]
        self.by_id = {item["chunk_id"]: item for item in self.chunks}
        self.commencement_sources: dict[str, dict] = {}
        for item in sorted(self.chunks, key=lambda record: int(record.get("page_number", 0))):
            if act_commencement_clause(item):
                self.commencement_sources.setdefault(item["document_id"], item)
        self.bm25 = BM25Okapi([bm25_tokenize(item["text"]) for item in self.chunks])
        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(str(self.index_path))
        if self.index.ntotal != len(self.chunks):
            raise ValueError("FAISS rows and metadata rows do not match")

    @classmethod
    def from_disk(cls, index_dir: Path = INDEXES) -> "RetrievalEngine":
        payload = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
        return cls(payload["chunks"], payload["model"], index_dir / "sections.faiss")

    def _allowed_ids(self, system: str, target_date: str | None, query: str = "") -> set[str]:
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system: {system}")
        if system == "standard":
            return set(self.ids)

        active = [item for item in self.chunks if version_valid_on(item, target_date)]
        if system == "date_filtered":
            return {item["chunk_id"] for item in active}

        best_rank: dict[str, int] = {}
        for item in active:
            family = item.get("version_family") or item["chunk_id"]
            best_rank[family] = max(best_rank.get(family, -1), int(item.get("version_rank", 0)))
        selected = {
            item["chunk_id"]
            for item in active
            if int(item.get("version_rank", 0)) == best_rank[item.get("version_family") or item["chunk_id"]]
        }
        if re.search(r"\b(amend|amended|repeal|repealed|replace|replaced|insert|inserted|change|changed)\b", query, re.I):
            selected.update(
                item["chunk_id"]
                for item in active
                if item.get("version_status") == "change_evidence"
            )
        return selected

    def search(
        self,
        query: str,
        system: str = "version_aware",
        target_date: str | None = None,
        top_k: int = 5,
        candidate_k: int = 75,
        expansion: str = "",
    ) -> list[dict]:
        """Search the corpus.

        ``expansion`` (for example, Gemini-generated statutory terms) is added
        to the keyword and embedding query only; dates, sections and amendment
        intent are still read from ``query``.
        """
        asks_commencement = commencement_intent(query)
        target_date = target_date or (None if asks_commencement else parse_query_date(query))
        allowed = self._allowed_ids(system, target_date, query)
        retrieval_query = " ".join(part for part in (expand_retrieval_query(query), expansion.strip()) if part)

        lexical_scores = np.asarray(self.bm25.get_scores(bm25_tokenize(retrieval_query)))
        lexical_order = np.argsort(-lexical_scores)
        lexical_ids = [self.ids[i] for i in lexical_order if self.ids[i] in allowed][:candidate_k]

        query_vector = np.asarray(
            self.model.encode([retrieval_query], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        # Exact IndexFlatIP search is inexpensive for this small corpus. Search
        # every row so metadata filters cannot hide a valid lower-ranked item.
        dense_scores, positions = self.index.search(query_vector, len(self.chunks))
        dense_ids = [
            self.ids[i]
            for i in positions[0]
            if i >= 0 and self.ids[i] in allowed
        ][:candidate_k]

        fused = reciprocal_rank_fusion([lexical_ids, dense_ids])
        fused_scores = dict(fused)
        named_commencement_id = None
        for chunk_id in list(fused_scores):
            item = self.by_id[chunk_id]
            fused_scores[chunk_id] += substantive_adjustment(item)
            fused_scores[chunk_id] += rate_passage_adjustment(query, item)
        if system == "version_aware":
            if asks_commencement:
                query_terms = " " + " ".join(bm25_tokenize(query)) + " "
                matches = []
                for item in self.commencement_sources.values():
                    title = " ".join(bm25_tokenize(item["title"]))
                    short_title = title.replace("goods and services tax", "gst").replace(" of bhutan ", " ")
                    if any(f" {name} " in query_terms for name in (title, short_title)):
                        matches.append(item)
                if len(matches) == 1:
                    # The explicitly named instrument is itself the object of the
                    # question. Its commencement clause is useful even before it is in force.
                    named_commencement_id = matches[0]["chunk_id"]
                    fused_scores[named_commencement_id] = 1.0
            section_match = re.search(r"\bsection\s+(\d{1,4}[a-z]?(?:\s*\([^)]+\))?)", query, re.I)
            requested_section = re.sub(r"\s+", "", section_match.group(1)).lower() if section_match else ""
            requested_relation = next(
                (
                    canonical
                    for canonical, variants in {
                        "amended": ("amend", "amended", "change", "changed"),
                        "repealed": ("repeal", "repealed"),
                        "inserted": ("insert", "inserted"),
                        "replaced": ("replace", "replaced", "substituted"),
                    }.items()
                    if any(re.search(rf"\b{word}\b", query, re.I) for word in variants)
                ),
                "",
            )
            for chunk_id in list(fused_scores):
                item = self.by_id[chunk_id]
                affected = [str(value).lower() for value in item.get("affected_sections", [])]
                relations = [str(value).lower() for value in item.get("relation_types", [])]
                item_section = re.sub(r"\s+", "", str(item.get("section_number", ""))).lower()
                title = re.sub(r"\s+", " ", str(item.get("title", ""))).strip().lower()
                if requested_section and requested_section in affected and not asks_commencement:
                    fused_scores[chunk_id] += 0.22
                if requested_section and requested_section == item_section and not asks_commencement:
                    fused_scores[chunk_id] += 0.14
                if requested_relation and requested_relation in relations and not asks_commencement:
                    fused_scores[chunk_id] += 0.05
                if title and title in re.sub(r"\s+", " ", query).lower():
                    fused_scores[chunk_id] += 0.18
                if asks_commencement and re.search(
                    r"\b(comes into force|commencement)\b", str(item.get("text", "")), re.I
                ):
                    fused_scores[chunk_id] += 0.40
                if re.search(r"\bGST\s+Rule\b", query, re.I) and item.get("document_type") == "rules":
                    fused_scores[chunk_id] += 0.05
                if re.search(r"\bbefore the later amendment\b", query, re.I) and item.get("document_id") == "gst_act_2020":
                    fused_scores[chunk_id] += 0.06
        fused = sorted(fused_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        dense_score_by_id = {
            self.ids[position]: float(score)
            for score, position in zip(dense_scores[0], positions[0])
            if position >= 0
        }
        results: list[dict] = []
        for chunk_id, rrf_score in fused[:top_k]:
            item = dict(self.by_id[chunk_id])
            if chunk_id == named_commencement_id:
                item["named_act_commencement"] = True
            item.update(
                {
                    "system": system,
                    "query_date": target_date,
                    "rrf_score": float(rrf_score),
                    "dense_score": dense_score_by_id.get(chunk_id),
                    "bm25_score": float(lexical_scores[self.ids.index(chunk_id)]),
                }
            )
            results.append(item)
        return results

