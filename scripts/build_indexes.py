from __future__ import annotations

import json
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from gst_rag.paths import INDEXES, MODELS, PROCESSED, RESULTS
from gst_rag.utils import read_jsonl


def main() -> None:
    sections = read_jsonl(PROCESSED / "sections.jsonl")
    metrics_path = RESULTS / "minilm_metrics.json"
    selected_model = "sentence-transformers/all-MiniLM-L6-v2"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("selection_rule_result") == "fine_tuned" and (MODELS / "minilm-bhutan-gst").exists():
            selected_model = str(MODELS / "minilm-bhutan-gst")
    model = SentenceTransformer(selected_model)
    embeddings = np.asarray(model.encode([s["text"] for s in sections], normalize_embeddings=True, show_progress_bar=True), dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    INDEXES.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEXES / "sections.faiss"))
    (INDEXES / "index_metadata.json").write_text(json.dumps({"model": selected_model, "chunks": sections}, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(sections)} chunks using {selected_model}")


if __name__ == "__main__":
    main()
