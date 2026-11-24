from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Dataset"
RAW_PDFS = DATASET / "raw_pdfs"
EXTRACTED = DATASET / "extracted_text"
PROCESSED = DATASET / "processed_sections"
METADATA = DATASET / "metadata"
TRAINING = DATASET / "training_examples"
BENCHMARK = DATASET / "benchmark"
TRANSCRIPTIONS = DATASET / "manual_transcriptions"
MODELS = ROOT / "models"
INDEXES = ROOT / "indexes"
RESULTS = ROOT / "results"
MANIFEST = METADATA / "corpus_manifest.csv"


def ensure_directories() -> None:
    for path in (RAW_PDFS, EXTRACTED, PROCESSED, METADATA, TRAINING, BENCHMARK, TRANSCRIPTIONS, MODELS, INDEXES, RESULTS):
        path.mkdir(parents=True, exist_ok=True)

