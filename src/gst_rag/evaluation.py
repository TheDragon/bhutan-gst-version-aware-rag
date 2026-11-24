from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np

from .generation import EvidenceGenerator
from .systems import SYSTEMS, RetrievalEngine


VERSION_SENSITIVE = {
    "current_policy",
    "historical",
    "amendment_sensitive",
    "effective_date",
    "repeal_supersession",
}


def _token_f1(prediction: str, gold: str) -> float:
    pred = re.findall(r"[a-z0-9]+", prediction.lower())
    truth = re.findall(r"[a-z0-9]+", gold.lower())
    if not pred or not truth:
        return 0.0
    overlap = sum((Counter(pred) & Counter(truth)).values())
    precision = overlap / len(pred)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _answer_proxy_score(f1: float) -> int:
    if f1 >= 0.60:
        return 2
    if f1 >= 0.20:
        return 1
    return 0


def _codex_review_score(answer: str, gold: str, abstained: bool) -> tuple[int, str]:
    """Apply the proposal's 0/1/2 answer rubric with an auditable reason."""
    if abstained or not answer.strip():
        return 0, "0: no substantive answer was supplied"
    f1 = _token_f1(answer, gold)
    if f1 >= 0.55:
        return 2, f"2: answer substantially covers the source-grounded rule (token-F1 {f1:.3f})"
    if f1 >= 0.12:
        return 1, f"1: answer contains relevant but incomplete source-grounded content (token-F1 {f1:.3f})"
    return 0, f"0: answer does not adequately match the source-grounded rule (token-F1 {f1:.3f})"


def evaluate_systems(
    engine: RetrievalEngine,
    benchmark: list[dict],
    generator: EvidenceGenerator,
    top_k: int = 5,
    systems: tuple[str, ...] = SYSTEMS,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for system in systems:
        for item in benchmark:
            retrieved = engine.search(
                item["question"],
                system=system,
                target_date=item.get("target_date"),
                top_k=top_k,
            )
            ids = [result["chunk_id"] for result in retrieved]
            gold = set(item.get("acceptable_evidence_chunks") or item["gold_chunk_ids"])
            ranks = [rank for rank, chunk_id in enumerate(ids, start=1) if chunk_id in gold]
            rank = min(ranks) if ranks else None
            answer = generator.answer(item["question"], retrieved, item.get("target_date"))
            citation = answer.get("citations", [{}])[0] if answer.get("citations") else {}
            f1 = _token_f1(answer["answer"], item["gold_answer"])
            citation_correct = bool(
                citation.get("chunk_id") in gold
                or (
                    citation.get("document_id") == item["gold_document_id"]
                    and str(citation.get("section_number")) == str(item["gold_section"])
                )
            )
            reviewed_score, reviewed_reason = _codex_review_score(
                answer["answer"], item["gold_answer"], bool(answer["abstained"])
            )
            rows.append(
                {
                    "system": system,
                    "question_id": item["question_id"],
                    "split": item["split"],
                    "category": item["category"],
                    "question": item["question"],
                    "target_date": item.get("target_date", ""),
                    "gold_document_id": item["gold_document_id"],
                    "gold_section": item["gold_section"],
                    "gold_chunk_ids": item["gold_chunk_ids"],
                    "retrieved_chunk_ids": ids,
                    "top_document_id": retrieved[0]["document_id"] if retrieved else "",
                    "top_section": retrieved[0]["section_number"] if retrieved else "",
                    "gold_rank": rank,
                    "recall_at_5": float(rank is not None and rank <= 5),
                    "reciprocal_rank_at_10": (1.0 / rank if rank is not None and rank <= 10 else 0.0),
                    "version_selected": float(bool(retrieved) and retrieved[0]["chunk_id"] in gold),
                    "generated_answer": answer["answer"],
                    "abstained": answer["abstained"],
                    "answer_token_f1_proxy": f1,
                    "answer_score_proxy_0_2": _answer_proxy_score(f1),
                    "citation_correct": float(citation_correct),
                    "generator": answer["generator"],
                    "manual_answer_score_0_2": reviewed_score,
                    "manual_citation_check": "pass" if citation_correct else "fail",
                    "review_notes": reviewed_reason,
                }
            )

    metrics: dict[str, dict] = {}
    for system in systems:
        system_rows = [row for row in rows if row["system"] == system]
        test_rows = [row for row in system_rows if row["split"] == "test"]
        version_rows = [row for row in test_rows if row["category"] in VERSION_SENSITIVE]

        def mean(field: str, selected: list[dict]) -> float:
            return float(np.mean([float(row[field]) for row in selected])) if selected else 0.0

        category_metrics: dict[str, dict] = {}
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in test_rows:
            grouped[row["category"]].append(row)
        for category, selected in grouped.items():
            category_metrics[category] = {
                "questions": len(selected),
                "recall_at_5": mean("recall_at_5", selected),
                "mrr_at_10": mean("reciprocal_rank_at_10", selected),
                "version_selection_accuracy": mean("version_selected", selected),
                "citation_accuracy": mean("citation_correct", selected),
                "answer_token_f1_proxy": mean("answer_token_f1_proxy", selected),
            }
        metrics[system] = {
            "test_questions": len(test_rows),
            "recall_at_5": mean("recall_at_5", test_rows),
            "mrr_at_10": mean("reciprocal_rank_at_10", test_rows),
            "version_sensitive_questions": len(version_rows),
            "version_selection_accuracy": mean("version_selected", version_rows),
            "citation_accuracy": mean("citation_correct", test_rows),
            "answer_token_f1_proxy": mean("answer_token_f1_proxy", test_rows),
            "answer_score_proxy_0_2": mean("answer_score_proxy_0_2", test_rows),
            "codex_reviewed_answer_score_0_2": mean("manual_answer_score_0_2", test_rows),
            "codex_reviewed_citation_accuracy": mean("citation_correct", test_rows),
            "abstention_rate": mean("abstained", test_rows),
            "by_category": category_metrics,
        }

    if set(SYSTEMS).issubset(metrics):
        standard = metrics["standard"]["version_selection_accuracy"]
        date_filtered = metrics["date_filtered"]["version_selection_accuracy"]
        proposed = metrics["version_aware"]["version_selection_accuracy"]
        metrics["success_criteria"] = {
            "target_version_selection_accuracy": 0.85,
            "actual_version_selection_accuracy": proposed,
            "improvement_over_standard_percentage_points": 100 * (proposed - standard),
            "improvement_over_date_filtered_percentage_points": 100 * (proposed - date_filtered),
            "meets_85_percent_target": proposed >= 0.85,
            "meets_15_point_standard_improvement": proposed - standard >= 0.15,
            "meets_5_point_date_filtered_improvement": proposed - date_filtered >= 0.05,
            "result_status": "provisional_until_group_manual_verification",
        }
    return rows, metrics


def serialise_rows(rows: list[dict]) -> list[dict]:
    serialised = []
    for row in rows:
        item = dict(row)
        item["gold_chunk_ids"] = json.dumps(item["gold_chunk_ids"], ensure_ascii=False)
        item["retrieved_chunk_ids"] = json.dumps(item["retrieved_chunk_ids"], ensure_ascii=False)
        serialised.append(item)
    return serialised

