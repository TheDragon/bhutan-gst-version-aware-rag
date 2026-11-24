from app.backend.main import documents, health, root


def test_root_describes_api():
    assert root()["docs"] == "/docs"


def test_health_reports_index():
    payload = health()
    assert payload["status"] == "ready"
    assert payload["indexed_chunks"] == 1581
    assert payload["generator_model"] == "google/flan-t5-base"


def test_documents_are_available():
    records = documents()
    assert len(records) == 10
    assert sum(item["chunk_count"] for item in records) == 1581
