# Project Progress Through Controlled Evaluation

> Historical checkpoint. See the [current project README](README.md) for the application and the latest saved evaluation.

## Scope of this build

This build follows the project proposal through the controlled comparison stage. It intentionally excludes a web frontend and API backend.

## Proposal-to-implementation progress

| Proposal requirement | Current progress | Evidence |
|---|---|---|
| Official versioned corpus | Implemented with review flags | 10 available official PDFs, source URLs, hashes, extraction audit |
| OCR or fallback for failed extraction | Implemented with review flag | One-page scanned notice has a manual visual transcription because the PDF has no text layer |
| Section-aware chunks and provenance | Implemented | 1,581 chunks with document, section, page, dates, status, and source |
| Amendment and repeal relationships | Source-text verification implemented; legal review pending | 62 source-linked events |
| At least 200 MiniLM training examples | Implemented; human review pending | 300 generated triplets |
| MiniLM before/after comparison | Implemented | Recall@5: 0.521 to 0.813; MRR@10: 0.420 to 0.646 |
| Keyword and dense retrieval | Implemented | BM25, fine-tuned MiniLM, FAISS, and RRF |
| Three controlled systems | Implemented | Standard, date-filtered, and version-aware retrieval use the same base components |
| Two generator candidates | Implemented on development split | FLAN-T5-small and FLAN-T5-base; base selected |
| 120-question benchmark | Structurally implemented; group verification pending | 20 development and 100 test questions in six proposed categories |
| Controlled comparison | Implemented provisionally | Retrieval, version selection, citation, answer proxy, and abstention results saved |
| Manual answer accuracy | Not complete | The prediction CSV contains blank manual scoring columns for group review |

## Corpus and preprocessing results

- 10 official PDF records are now available.
- The official 2026 Amendment Act was located on the Ministry of Finance website and added to the corpus.
- The scanned December 2025 implementation notice was visually transcribed and remains marked for group verification.
- The rebuilt corpus contains 1,581 searchable chunks.
- Each chunk records provenance and version fields, including `version_family`, `version_start`, `version_end`, `version_rank`, and linked amendment events.

## MiniLM result retained from the mid-review

| Metric | Pretrained | Fine-tuned |
|---|---:|---:|
| Recall@5 | 0.521 | 0.813 |
| MRR@10 | 0.420 | 0.646 |

These results use the 48-item development split from the generated retriever-training data. They are separate from the final 120-question benchmark.

## Generator selection

Two pretrained instruction models were compared on the 20-question development benchmark:

| Model | Mean answer token-F1 proxy |
|---|---:|
| `google/flan-t5-small` | 0.029 |
| `google/flan-t5-base` | 0.284 |

FLAN-T5-base was selected and then kept fixed across all three systems. This automated proxy does not replace manual answer scoring.

## Provisional controlled comparison on 100 test questions

| System | Recall@5 | MRR@10 | Version Selection Accuracy | Citation Accuracy | Answer token-F1 proxy | Abstention rate |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 0.410 | 0.315 | 0.293 | 0.290 | 0.140 | 0.250 |
| Date-filtered | 0.470 | 0.382 | 0.370 | 0.370 | 0.158 | 0.330 |
| Version-aware | 0.840 | 0.751 | 0.696 | 0.450 | 0.161 | 0.290 |

The version-aware system improved Version Selection Accuracy by 40.2 percentage points over standard RAG and 32.6 points over date-filtered RAG. It passed the proposal's relative-improvement targets but did not meet the absolute 85% target.

## Interpretation

The experiment currently supports the claim that explicit amendment metadata helps retrieval and version selection. It does not yet support a final claim about legal answer accuracy. The benchmark was source-grounded programmatically and must still be reviewed manually. The selected generator also produced weak answer-overlap scores and sometimes abstained, so answer generation remains an important bottleneck.

## Required group review before final reporting

1. Check the scanned-notice transcription against the PDF image.
2. Verify all 62 amendment relationships and effective dates.
3. Review the 300 MiniLM training triplets.
4. Review all 120 benchmark questions, answers, dates, documents, sections, and acceptable evidence.
5. Complete the blank manual answer score and citation check columns in `results/controlled_comparison_predictions.csv`.
6. Rerun the summary after corrections and report both successful and unsuccessful criteria.

## Main runnable notebook

Use `notebooks/05_full_project_to_controlled_evaluation.ipynb`. It includes the earlier mid-review progress and every new stage through the controlled comparison. Expensive rebuild cells are disabled by default so the notebook opens quickly; enable them only when intentionally rebuilding models and results.

