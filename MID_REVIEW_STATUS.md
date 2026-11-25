# Mid-Review Status

> Historical checkpoint. See the [current project README](README.md) for GST Lens, setup, and the latest saved evaluation.

## Completed

- Reproducible project structure and Jupyter workflow
- Official DRC corpus manifest with URLs, file hashes, and availability status
- Nine downloaded official PDFs
- English-page extraction audit
- 1,569 section/page evidence chunks with source provenance
- 61 automatically detected amendment candidates marked for human review
- 300 machine-generated MiniLM training triples marked for human review
- One-epoch CPU fine-tuning experiment
- BM25 + fine-tuned MiniLM + FAISS hybrid retrieval prototype
- Date filtering using known document-level effective dates
- Automated checkpoint validator and unit tests

## Development results

| Metric | Pretrained MiniLM | Fine-tuned MiniLM |
|---|---:|---:|
| Recall@5 | 0.521 | 0.813 |
| MRR@10 | 0.420 | 0.646 |

The fine-tuned model passed the proposal's development selection rule because neither metric decreased. These results use 48 held-out machine-generated development examples; they are not final benchmark results.

## Must be checked by the group

1. Verify the 61 amendment relationships against the official Acts.
2. Review and correct the generated training questions before treating them as research-quality annotations.
3. Obtain the official GST Amendment Act of Bhutan 2026. The DRC June 2026 notice refers to it, but the Act PDF was not located in the initial official-source search.
4. OCR or manually transcribe the image-only December 2025 GST implementation notice.

Until item 3 is resolved, the system must warn or abstain on claims about the exact post-June-2026 legal version.

## Mid-review demonstration

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\validate_checkpoint.py --checkpoint all
python scripts\demo_search.py "What does section 40 say about amending a GST return in 2025?" --top-k 3
python -m jupyter lab
```

This prototype retrieves official evidence. It does not provide legal advice.

