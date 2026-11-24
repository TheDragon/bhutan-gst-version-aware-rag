from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from gst_rag.benchmark import build_benchmark
from gst_rag.corpus import build_sections, extract_amendment_events, load_manifest
from gst_rag.paths import BENCHMARK, METADATA, PROCESSED, RESULTS, TRAINING, ensure_directories
from gst_rag.quality import substantive_legal_text, toc_like
from gst_rag.training_data import generate_training_examples
from gst_rag.utils import read_jsonl, write_jsonl
from gst_rag.versioning import enrich_section_versions, verify_amendment_events


REVIEW_DIR = RESULTS / "review"


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def main() -> None:
    ensure_directories()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    base_sections = (
        read_jsonl(PROCESSED / "sections_base.jsonl")
        if (PROCESSED / "sections_base.jsonl").exists()
        else build_sections()
    )

    detected = extract_amendment_events(base_sections)
    events = verify_amendment_events(base_sections, detected, load_manifest())
    events.to_csv(METADATA / "amendment_events.csv", index=False, encoding="utf-8")
    events.to_csv(REVIEW_DIR / "amendment_relationship_review.csv", index=False, encoding="utf-8")

    sections = enrich_section_versions(base_sections, events)
    write_jsonl(PROCESSED / "sections.jsonl", sections)
    by_id = {item["chunk_id"]: item for item in sections}

    triplets = generate_training_examples(sections, minimum=300, maximum=300)
    write_jsonl(TRAINING / "minilm_training.jsonl", triplets)
    triplet_rows = []
    for item in triplets:
        positive = by_id[item["positive_chunk_id"]]
        negative = by_id[item["negative_chunk_id"]]
        triplet_rows.append(
            {
                "example_id": item["example_id"],
                "positive_chunk_id": item["positive_chunk_id"],
                "negative_chunk_id": item["negative_chunk_id"],
                "positive_substantive": substantive_legal_text(item["positive_text"]),
                "positive_is_toc": toc_like(item["positive_text"]),
                "positive_negative_distinct": _normalise(item["positive_text"]) != _normalise(item["negative_text"]),
                "different_section": str(positive["section_number"]) != str(negative["section_number"]),
                "negative_similarity": item.get("negative_similarity", ""),
                "review_status": item["verification_status"],
                "review_notes": item["review_notes"],
            }
        )
    pd.DataFrame(triplet_rows).to_csv(
        REVIEW_DIR / "training_triplet_review.csv", index=False, encoding="utf-8"
    )

    benchmark = build_benchmark(sections, events)
    write_jsonl(BENCHMARK / "gst_120_benchmark.jsonl", benchmark)
    benchmark_rows = []
    for item in benchmark:
        evidence = [by_id[chunk_id] for chunk_id in item["gold_chunk_ids"] if chunk_id in by_id]
        benchmark_rows.append(
            {
                "question_id": item["question_id"],
                "split": item["split"],
                "category": item["category"],
                "question": item["question"],
                "target_date": item["target_date"],
                "gold_document_id": item["gold_document_id"],
                "gold_section": item["gold_section"],
                "acceptable_evidence_count": len(evidence),
                "all_evidence_substantive": bool(evidence) and all(substantive_legal_text(x["text"]) for x in evidence),
                "contains_toc_evidence": any(toc_like(x["text"]) for x in evidence),
                "review_status": item["verification_status"],
                "review_notes": item["review_notes"],
            }
        )
    pd.DataFrame(benchmark_rows).to_csv(
        REVIEW_DIR / "benchmark_question_review.csv", index=False, encoding="utf-8"
    )

    summary = {
        "amendment_relationships": len(events),
        "amendment_relationships_source_verified": int(
            (events["verification_status"] == "codex_source_verified").sum()
        ),
        "training_triplets": len(triplets),
        "training_triplets_passed": sum(
            row["positive_substantive"]
            and not row["positive_is_toc"]
            and row["positive_negative_distinct"]
            and row["different_section"]
            for row in triplet_rows
        ),
        "benchmark_questions": len(benchmark),
        "benchmark_questions_passed": sum(
            row["all_evidence_substantive"] and not row["contains_toc_evidence"]
            for row in benchmark_rows
        ),
        "unique_benchmark_questions": len({item["question"] for item in benchmark}),
        "final_human_signoff_deferred": True,
    }
    (REVIEW_DIR / "review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
