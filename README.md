# Version-Aware RAG for Bhutan GST

Research implementation of a version-aware retrieval pipeline for official Bhutan GST legislation. The repository now includes the earlier mid-review checkpoint plus a provisional three-system controlled comparison. The frontend and API are intentionally deferred.

## Mid-review scope

1. Freeze and audit an official-source corpus.
2. Extract English text from the PDFs.
3. Create page/section-aware chunks with provenance metadata.
4. Extract candidate amendment relationships for human review.
5. Create at least 200 weakly supervised question-passage training triples.
6. Compare pretrained and fine-tuned MiniLM retrieval.
7. Build BM25, FAISS, and a small version-filtered retrieval demonstration.

This is an academic information-retrieval prototype, not legal advice. Automatically extracted effective dates and amendment relationships must be manually checked against the official documents before being presented as verified legal facts.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m jupyter lab
```

The project pins Sentence Transformers below version 4 because newer multimodal releases may require TorchCodec/FFmpeg even though this project only trains a text embedding model.

If an old global `jupyter_contrib_nbextensions` installation breaks `nbconvert`, notebooks can still be tested without optional plugins:

```powershell
python scripts/execute_notebook.py notebooks/02_versioned_sections.ipynb --output results/notebook_runs/02_versioned_sections.ipynb
```

The notebooks in `notebooks/` can be run in order. The same reproducible pipeline is also available from the command line:

```powershell
python scripts/run_to_mid_review.py --skip-training
python scripts/train_minilm.py --epochs 1
python scripts/build_indexes.py
python scripts/validate_checkpoint.py --checkpoint all
```

Use `--skip-training` for a fast corpus checkpoint. MiniLM fine-tuning is CPU-compatible but may take several minutes.

## Corpus cut-off

The initial corpus cut-off is **2026-09-30**. The DRC June 2026 implementation notice refers to a GST Amendment Act of Bhutan 2026, but an official downloadable copy was not located during initial collection. This is tracked in `Dataset/metadata/corpus_manifest.csv` as a blocking provenance gap for claims about the post-June-2026 legal position.

## Outputs

- `Dataset/metadata/corpus_manifest.csv`: source and version register
- `Dataset/metadata/extraction_report.csv`: extraction audit
- `Dataset/metadata/amendment_events.csv`: candidate relationships requiring review
- `Dataset/processed_sections/sections.jsonl`: retrievable evidence chunks
- `Dataset/training_examples/minilm_training.jsonl`: retriever training triples
- `results/checkpoint_report.json`: validation summary
- `results/minilm_metrics.json`: baseline/fine-tuned retrieval metrics

## Current mid-review checkpoint

- 9 official PDFs downloaded and hash-audited
- 8 PDFs with usable English text; 1 image-only notice flagged for OCR
- 1,569 retrievable chunks
- 61 candidate amendment events requiring human verification
- 300 generated training triples requiring sample-by-sample review before final research use
- MiniLM development Recall@5: 0.521 before fine-tuning, 0.813 after one epoch
- MiniLM development MRR@10: 0.420 before fine-tuning, 0.646 after one epoch
- FAISS index built for all chunks and hybrid BM25+dense retrieval smoke-tested

These development figures are not the proposal's final benchmark results. The 120-question benchmark is deliberately reserved for the later evaluation phase.

## Progress after the mid-review

- Official GST Amendment Act 2026 added; all 10 manifest records are now available.
- Scanned implementation notice transcribed with an explicit group-review flag.
- 1,581 version-enriched chunks and 62 source-linked amendment events.
- Standard, date-filtered, and version-aware retrieval systems implemented.
- Proposal-shaped 120-question benchmark created with 20 development and 100 test questions; all items remain marked for group review.
- FLAN-T5-small and FLAN-T5-base compared; FLAN-T5-base selected on the development split.
- Provisional test Version Selection Accuracy: standard 0.293, date-filtered 0.370, version-aware 0.696.

See `PROJECT_PROGRESS_TO_STEP_6.md` and `notebooks/05_full_project_to_controlled_evaluation.ipynb` for the complete workflow and limitations.

## GST Lens web application

The research pipeline now has a FastAPI backend and a responsive React
frontend under `app/`. It provides five connected workspaces:

- project overview and pipeline statistics;
- dated question answering with an evidence inspector;
- side-by-side comparison of all three retrieval systems;
- section version timelines;
- an indexed official-document library.

Install and run:

```powershell
cd app\frontend
npm install
cd ..\..
.\scripts\run_research_app.ps1
```

Use `-FastDemo` to substitute deterministic extractive generation while
demonstrating the interface. Without it, the backend uses the selected
`google/flan-t5-base` generator.
