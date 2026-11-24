from __future__ import annotations

import argparse
import json

from gst_rag.corpus import (
    build_sections,
    corpus_summary,
    download_corpus,
    extract_amendment_events,
    extract_corpus,
)
from gst_rag.corpus import load_manifest
from gst_rag.paths import METADATA, PROCESSED, RESULTS, TRAINING, ensure_directories
from gst_rag.training_data import generate_training_examples
from gst_rag.versioning import enrich_section_versions, verify_amendment_events
from gst_rag.utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="Skip model fine-tuning, not training-data creation")
    args = parser.parse_args()
    ensure_directories()
    if not args.skip_download:
        audit = download_corpus()
        print(f"Download checkpoint: {sum(audit.download_status.isin(['downloaded', 'existing']))} PDFs available")
    report = extract_corpus()
    print(f"Extraction checkpoint: {sum(report.status.astype(str).str.startswith('extracted'))} documents extracted")
    sections = build_sections()
    write_jsonl(PROCESSED / "sections_base.jsonl", sections)
    events = extract_amendment_events(sections)
    events = verify_amendment_events(sections, events, load_manifest())
    events.to_csv(METADATA / "amendment_events.csv", index=False, encoding="utf-8")
    sections = enrich_section_versions(sections, events)
    write_jsonl(PROCESSED / "sections.jsonl", sections)
    print(f"Versioning checkpoint: {len(sections)} chunks and {len(events)} candidate events")
    examples = generate_training_examples(sections)
    write_jsonl(TRAINING / "minilm_training.jsonl", examples)
    print(f"Training-data checkpoint: {len(examples)} examples")
    summary = corpus_summary(sections) | {"candidate_amendment_events": len(events), "training_examples": len(examples)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not args.skip_training:
        print("Corpus pipeline complete. Run scripts/train_minilm.py for the measured model comparison.")


if __name__ == "__main__":
    main()

