from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gst_rag.benchmark import build_benchmark
from gst_rag.evaluation import evaluate_systems, serialise_rows
from gst_rag.generation import EvidenceGenerator
from gst_rag.paths import BENCHMARK, METADATA, PROCESSED, RESULTS, ensure_directories
from gst_rag.systems import RetrievalEngine
from gst_rag.utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="extractive-evidence-v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--fixed-benchmark", action="store_true", help="Use the existing benchmark file unchanged")
    parser.add_argument("--output-dir", type=Path, default=RESULTS, help="Directory for predictions and metrics")
    args = parser.parse_args()

    ensure_directories()
    benchmark_path = BENCHMARK / "gst_120_benchmark.jsonl"
    if args.fixed_benchmark:
        benchmark = read_jsonl(benchmark_path)
    else:
        sections = read_jsonl(PROCESSED / "sections.jsonl")
        events = pd.read_csv(METADATA / "amendment_events.csv").fillna("")
        benchmark = build_benchmark(sections, events)
    if len(benchmark) != 120:
        raise ValueError(f"Expected 120 benchmark questions, got {len(benchmark)}")
    if not args.fixed_benchmark:
        write_jsonl(benchmark_path, benchmark)

    selected = benchmark[: args.limit] if args.limit else benchmark
    engine = RetrievalEngine.from_disk()
    generator = EvidenceGenerator(args.generator)
    rows, metrics = evaluate_systems(engine, selected, generator, top_k=args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(serialise_rows(rows)).to_csv(
        args.output_dir / "controlled_comparison_predictions.csv", index=False, encoding="utf-8"
    )
    (args.output_dir / "controlled_comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

