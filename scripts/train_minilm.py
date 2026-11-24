from __future__ import annotations

import argparse
import json
import os
import random

# This project is text-only. Prevent unrelated global TensorFlow/Keras packages
# from being imported by Transformers in mixed Python environments.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from gst_rag.paths import MODELS, RESULTS, TRAINING
from gst_rag.utils import read_jsonl


BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def retrieval_metrics(model: SentenceTransformer, examples: list[dict]) -> dict:
    positives: dict[str, str] = {}
    for item in examples:
        positives[item["positive_chunk_id"]] = item["positive_text"]
    passage_ids = list(positives)
    passages = [positives[key] for key in passage_ids]
    passage_embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
    question_embeddings = model.encode([item["question"] for item in examples], normalize_embeddings=True, show_progress_bar=False)
    scores = np.asarray(question_embeddings) @ np.asarray(passage_embeddings).T
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for item, row in zip(examples, scores):
        order = np.argsort(-row)
        gold = passage_ids.index(item["positive_chunk_id"])
        rank = int(np.where(order == gold)[0][0]) + 1
        recalls.append(float(rank <= 5))
        reciprocal_ranks.append(1.0 / rank if rank <= 10 else 0.0)
    return {"recall_at_5": float(np.mean(recalls)), "mrr_at_10": float(np.mean(reciprocal_ranks)), "queries": len(examples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=240)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    records = read_jsonl(TRAINING / "minilm_training.jsonl")[: args.max_examples]
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Training data must include train and validation splits")
    model = SentenceTransformer(BASE_MODEL)
    before = retrieval_metrics(model, validation)
    samples = [InputExample(texts=[r["question"], r["positive_text"], r["negative_text"]]) for r in train]
    loader = DataLoader(samples, shuffle=True, batch_size=args.batch_size)
    loss = losses.TripletLoss(model=model, triplet_margin=0.35)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, warmup_steps=max(1, len(loader) // 10), show_progress_bar=True)
    output = MODELS / "minilm-bhutan-gst"
    model.save(str(output))
    after = retrieval_metrics(model, validation)
    selected = "fine_tuned" if after["recall_at_5"] >= before["recall_at_5"] and after["mrr_at_10"] >= before["mrr_at_10"] else "pretrained"
    metrics = {"base_model": BASE_MODEL, "epochs": args.epochs, "train_examples": len(train), "validation_examples": len(validation), "before": before, "after": after, "selection_rule_result": selected}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "minilm_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
