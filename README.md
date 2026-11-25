# Version-Aware RAG for Bhutan GST

GST Lens is a working research MVP for finding and comparing official Bhutan GST provisions as of a selected date. Its React frontend and FastAPI backend provide dated question answering, a citation and evidence inspector, side-by-side retrieval comparisons, section version timelines, and a searchable document library.

The corpus contains 10 official PDFs, including the 2026 GST Amendment Act, and 1,581 section-aware chunks. The corpus includes a transcription of one scanned implementation notice and has a 30 September 2026 cut-off. [Corpus manifest](Dataset/metadata/corpus_manifest.csv)

## Run GST Lens locally

Use Windows PowerShell from the repository root. The Python project supports **3.11 and 3.12**. The frontend lockfile requires **Node.js 20.19+ within Node 20, or Node.js 22.12+**.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
npm --prefix app/frontend ci
```

On a fresh clone, rebuild the corpus, train MiniLM on the 300 generated triplets, and build the FAISS index. The first run needs internet access for the source PDFs and pretrained model weights.

```powershell
.\.venv\Scripts\python.exe scripts/run_to_mid_review.py --skip-training
.\.venv\Scripts\python.exe scripts/train_minilm.py --epochs 1 --max-examples 300
.\.venv\Scripts\python.exe scripts/build_indexes.py
```

Then start the app and open [GST Lens on localhost](http://127.0.0.1:5173).

```powershell
.\scripts\run_research_app.ps1 -FastDemo
```

`-FastDemo` uses extractive answers. For FLAN-T5-base answer generation, launch without that flag from a fresh PowerShell session. The backend listens on port 8000. Press Ctrl+C to stop the launcher.

The GitHub clone includes the source code, notebooks, corpus manifest, and benchmark file. Raw corpus PDFs, extracted passages, generated triplets, model weights, FAISS files, and evaluation outputs are stored locally and excluded by Git.

## Retrieval and evaluation

The three retrieval modes share BM25 keyword search, MiniLM embeddings searched with FAISS, and reciprocal rank fusion. Standard RAG searches the corpus, date-filtered RAG filters by effective dates, and version-aware RAG also uses amendment relationships and version metadata.

On the 60-example embedding validation split, one epoch of MiniLM fine-tuning improved Recall@5 from **56.7% to 90.0%** and MRR@10 from **0.465 to 0.838**. This is a separate experiment from the controlled RAG comparison.

The latest saved controlled comparison uses the revised 120-question benchmark (20 development and 100 test questions) and the extractive generator. Each system retrieved ten passages per question for MRR@10. Version selection was measured on the 92 version-sensitive test questions.

| System | Recall@5 | MRR@10 | Version-selection accuracy | Automated citation match |
| --- | ---: | ---: | ---: | ---: |
| Standard RAG | 56% | 0.421 | 26.1% | 30% |
| Date-filtered RAG | 57% | 0.467 | 34.8% | 38% |
| Version-aware RAG | 97% | 0.946 | 94.6% | 90% |

Version-aware RAG met the proposal's **85%** target on this benchmark, selecting the expected passage first for **87 of 92** version-sensitive questions. A diagnostic on the same test questions before the named-Act commencement lookup scored **70 of 92 (76.1%)**. The 17 effective-date test questions share one source clause and name the Act in the question, so this result measures performance on the project benchmark. It does not establish performance on unseen Acts or provision-specific commencement conditions.

Citation match measures agreement with the benchmark evidence labels. Use the evidence inspector to see the cited passage and assess whether it supports the answer.

To reproduce this comparison using the existing benchmark, run the command below. It saves predictions and metrics under `results/commencement_evaluation/`.

```powershell
.\.venv\Scripts\python.exe scripts/run_final_experiment.py --fixed-benchmark --top-k 10 --output-dir results/commencement_evaluation --generator extractive-evidence-v1
```

## Project materials and historical checkpoints

- [Mid-review checkpoint](MID_REVIEW_STATUS.md) and [technical report](docs/MID_REVIEW_TECHNICAL_REPORT.md) document the original corpus and retriever.
- [Progress through controlled evaluation](PROJECT_PROGRESS_TO_STEP_6.md) and the [full workflow notebook](notebooks/05_full_project_to_controlled_evaluation.ipynb) describe the earlier experiment.
- [Post-mid-review technical brief](output/pdf/Bhutan_GST_RAG_Post_Mid_Review_Documentation.pdf) documents the earlier results and application handoff.
- [Application architecture](app/ARCHITECTURE.md) describes the frontend and API.
