from __future__ import annotations

import gc
import json

import numpy as np

from gst_rag.evaluation import _token_f1
from gst_rag.generation import EvidenceGenerator
from gst_rag.paths import BENCHMARK, RESULTS
from gst_rag.systems import RetrievalEngine
from gst_rag.utils import read_jsonl


CANDIDATES = ["google/flan-t5-small", "google/flan-t5-base"]


def main() -> None:
    benchmark = [item for item in read_jsonl(BENCHMARK / "gst_120_benchmark.jsonl") if item["split"] == "development"]
    if len(benchmark) != 20:
        raise ValueError(f"Expected 20 development questions, found {len(benchmark)}")
    engine = RetrievalEngine.from_disk()
    evidence = {
        item["question_id"]: engine.search(
            item["question"], "version_aware", item["target_date"], top_k=5
        )
        for item in benchmark
    }

    results: dict[str, dict] = {}
    for model_name in CANDIDATES:
        print(f"Evaluating {model_name}", flush=True)
        generator = EvidenceGenerator(model_name=model_name, max_new_tokens=64)
        scores: list[float] = []
        answers: list[dict] = []
        for item in benchmark:
            output = generator.answer(
                item["question"], evidence[item["question_id"]], item["target_date"]
            )
            score = _token_f1(output["answer"], item["gold_answer"])
            scores.append(score)
            answers.append(
                {
                    "question_id": item["question_id"],
                    "answer": output["answer"],
                    "token_f1_proxy": score,
                    "abstained": output["abstained"],
                }
            )
            print(f"  {item['question_id']}: {score:.3f}", flush=True)
        results[model_name] = {
            "development_questions": len(benchmark),
            "mean_answer_token_f1_proxy": float(np.mean(scores)),
            "abstention_rate": float(np.mean([row["abstained"] for row in answers])),
            "answers": answers,
        }
        del generator
        gc.collect()

    selected = max(CANDIDATES, key=lambda name: results[name]["mean_answer_token_f1_proxy"])
    payload = {
        "candidates": CANDIDATES,
        "selection_metric": "development_answer_token_f1_proxy",
        "selected_generator": selected,
        "results": results,
        "status": "development_model_selection_needs_manual_answer_review",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "generator_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()

