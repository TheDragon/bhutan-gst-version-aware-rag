from __future__ import annotations

import argparse
import json

import pandas as pd

from gst_rag.evaluation import evaluate_systems, serialise_rows
from gst_rag.generation import EvidenceGenerator
from gst_rag.paths import BENCHMARK, RESULTS
from gst_rag.systems import SYSTEMS, RetrievalEngine
from gst_rag.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=SYSTEMS)
    parser.add_argument("--generator", default="google/flan-t5-base")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=120)
    args = parser.parse_args()

    benchmark = read_jsonl(BENCHMARK / "gst_120_benchmark.jsonl")[args.start : args.end]
    engine = RetrievalEngine.from_disk()
    generator = EvidenceGenerator(args.generator, max_new_tokens=64)
    rows, metrics = evaluate_systems(
        engine, benchmark, generator, top_k=5, systems=(args.system,)
    )
    pd.DataFrame(serialise_rows(rows)).to_csv(
        RESULTS / f"generator_predictions_{args.system}_{args.start:03d}_{args.end:03d}.csv",
        index=False,
        encoding="utf-8",
    )
    (RESULTS / f"generator_metrics_{args.system}_{args.start:03d}_{args.end:03d}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

