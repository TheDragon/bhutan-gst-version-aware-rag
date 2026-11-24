from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gst_rag.paths import BENCHMARK, EXTRACTED, INDEXES, MANIFEST, METADATA, MODELS, PROCESSED, RAW_PDFS, RESULTS, TRAINING
from gst_rag.utils import read_jsonl


def validate_setup() -> dict:
    manifest = pd.read_csv(MANIFEST).fillna("")
    required = {"document_id", "title", "document_type", "source_url", "availability_status", "requires_manual_review"}
    assert required.issubset(manifest.columns)
    assert not manifest.document_id.duplicated().any()
    return {"manifest_rows": len(manifest), "known_missing_documents": int((manifest.availability_status != "available").sum())}


def validate_corpus() -> dict:
    manifest = pd.read_csv(MANIFEST).fillna("")
    available = manifest[manifest.availability_status == "available"]
    valid_pdfs = 0
    for filename in available.local_filename:
        path = RAW_PDFS / filename
        assert path.exists(), f"Missing PDF: {path}"
        assert path.read_bytes()[:4] == b"%PDF", f"Invalid PDF: {path}"
        valid_pdfs += 1
    report = pd.read_csv(METADATA / "extraction_report.csv").fillna("")
    extracted_mask = report.status.astype(str).str.startswith("extracted")
    assert int(extracted_mask.sum()) >= max(1, valid_pdfs - 1)
    return {"valid_pdfs": valid_pdfs, "extracted_documents": int(extracted_mask.sum()), "extracted_files": len(list(EXTRACTED.glob("*.jsonl")))}


def validate_midreview() -> dict:
    sections = read_jsonl(PROCESSED / "sections.jsonl")
    training = read_jsonl(TRAINING / "minilm_training.jsonl")
    events = pd.read_csv(METADATA / "amendment_events.csv").fillna("")
    assert len(sections) >= 100, f"Too few retrievable chunks: {len(sections)}"
    assert len(training) >= 200, f"Too few training examples: {len(training)}"
    assert len(events) >= 5, f"Too few candidate amendment events: {len(events)}"
    assert all(item["verification_status"] == "machine_generated_needs_review" for item in training)
    metrics = json.loads((RESULTS / "minilm_metrics.json").read_text(encoding="utf-8"))
    assert metrics["before"]["queries"] > 0
    assert metrics["after"]["recall_at_5"] >= metrics["before"]["recall_at_5"]
    assert metrics["after"]["mrr_at_10"] >= metrics["before"]["mrr_at_10"]
    assert metrics["selection_rule_result"] in {"fine_tuned", "pretrained"}
    assert (MODELS / "minilm-bhutan-gst").exists()
    assert (INDEXES / "sections.faiss").exists()
    assert (INDEXES / "index_metadata.json").exists()
    for notebook in Path("notebooks").glob("*.ipynb"):
        json.loads(notebook.read_text(encoding="utf-8"))
    return {
        "chunks": len(sections),
        "training_examples": len(training),
        "candidate_events": len(events),
        "notebooks": len(list(Path("notebooks").glob("*.ipynb"))),
        "minilm_selection": metrics["selection_rule_result"],
        "recall_at_5_before": metrics["before"]["recall_at_5"],
        "recall_at_5_after": metrics["after"]["recall_at_5"],
        "mrr_at_10_before": metrics["before"]["mrr_at_10"],
        "mrr_at_10_after": metrics["after"]["mrr_at_10"],
    }


def validate_final_research() -> dict:
    manifest = pd.read_csv(MANIFEST).fillna("")
    assert (manifest.availability_status == "available").all()
    sections = read_jsonl(PROCESSED / "sections.jsonl")
    assert all("version_family" in item and "version_rank" in item for item in sections)
    events = pd.read_csv(METADATA / "amendment_events.csv").fillna("")
    assert len(events) >= 62
    assert events.verification_status.astype(str).str.contains("group_review").all()

    benchmark = read_jsonl(BENCHMARK / "gst_120_benchmark.jsonl")
    assert len(benchmark) == 120
    assert sum(item["split"] == "development" for item in benchmark) == 20
    assert sum(item["split"] == "test" for item in benchmark) == 100
    assert all(item["requires_manual_review"] for item in benchmark)

    generator = json.loads((RESULTS / "generator_comparison.json").read_text(encoding="utf-8"))
    assert len(generator["candidates"]) == 2
    assert generator["selected_generator"] in generator["candidates"]
    comparison = json.loads((RESULTS / "controlled_comparison_metrics.json").read_text(encoding="utf-8"))
    assert {"standard", "date_filtered", "version_aware", "success_criteria"}.issubset(comparison)
    predictions = pd.read_csv(RESULTS / "controlled_comparison_predictions.csv").fillna("")
    assert len(predictions) == 360
    assert set(predictions.system) == {"standard", "date_filtered", "version_aware"}
    assert set(predictions.generator) == {generator["selected_generator"]}

    notebook = Path("notebooks/05_full_project_to_controlled_evaluation.ipynb")
    json.loads(notebook.read_text(encoding="utf-8"))
    return {
        "available_official_documents": int((manifest.availability_status == "available").sum()),
        "versioned_chunks": len(sections),
        "source_linked_events": len(events),
        "benchmark_questions": len(benchmark),
        "development_questions": 20,
        "test_questions": 100,
        "selected_generator": generator["selected_generator"],
        "standard_version_accuracy": comparison["standard"]["version_selection_accuracy"],
        "date_filtered_version_accuracy": comparison["date_filtered"]["version_selection_accuracy"],
        "version_aware_version_accuracy": comparison["version_aware"]["version_selection_accuracy"],
        "result_status": comparison["success_criteria"]["result_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["setup", "corpus", "midreview", "final", "all"], default="all")
    args = parser.parse_args()
    report: dict[str, dict] = {}
    if args.checkpoint in {"setup", "all"}:
        report["setup"] = validate_setup()
    if args.checkpoint in {"corpus", "all"}:
        report["corpus"] = validate_corpus()
    if args.checkpoint in {"midreview", "all"}:
        report["midreview"] = validate_midreview()
    if args.checkpoint in {"final", "all"}:
        report["final"] = validate_final_research()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "checkpoint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
