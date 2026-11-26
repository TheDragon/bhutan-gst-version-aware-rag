from datetime import date

from app.backend.main import SearchRequest, answer, documents, health, root
from gst_rag.quality import gst_definition_intent, gst_question_has_evidence, gst_rate_intent, normalize_user_question


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


def test_gst_definition_returns_applicable_official_sources():
    result = answer(SearchRequest(question="What is GST?", target_date=date(2026, 1, 1)))
    assert result["answer_strategy"] == "definition_lookup"
    assert result["answer"].startswith("GST stands for Goods and Services Tax.")
    assert "taxable supplies and imports" in result["answer"]
    assert [item["document_id"] for item in result["citations"]] == [
        "gst_consolidated_2025",
        "gst_implementation_notice_2025",
    ]
    assert [item["chunk_id"] for item in result["citations"]] == [item["chunk_id"] for item in result["evidence"]]

    historical = answer(SearchRequest(question="What does GST stand for?", target_date=date(2025, 12, 31)))
    assert historical["citations"][0]["document_id"] == "gst_act_2020"
    assert len(historical["citations"]) == 1
    assert not gst_definition_intent("What is the GST rate for a car?")


def test_short_tobacco_tax_query_asks_which_tax_instead_of_returning_unrelated_passages():
    result = answer(SearchRequest(question="cigarrete tax as of now", target_date=date(2026, 1, 1)))
    assert result["question"] == "cigarrete tax as of now"
    assert result["interpreted_question"] == "cigarette tax as of now"
    assert result["clarification_question"] == "cigarette GST as of now"
    assert result["reason"] == "clarify_tax_type"
    assert result["target_date"] == "2026-01-01"
    assert result["evidence"] == []
    assert gst_rate_intent(result["clarification_question"])
    assert normalize_user_question("Section 337(40) as of 2026-01-01") == "Section 337(40) as of 2026-01-01"

    excise = answer(SearchRequest(question="What is the excise tax on cigarettes?"))
    assert excise["reason"] == "clarify_tax_type"
    assert excise["clarification_question"] is None


def test_vague_questions_get_clarified_or_mapped_to_a_specific_intent():
    car_tax = answer(SearchRequest(question="tax on car"))
    assert car_tax["clarification_question"] == "GST on car"
    assert answer(SearchRequest(question="gst exemption"))["clarification_title"] == "What are you checking for an exemption?"
    assert answer(SearchRequest(question="section 40"))["clarification_title"] == "Which document is the section from?"
    assert gst_definition_intent("gst")
    assert gst_rate_intent("gst cigarette")
    assert "tax period" in normalize_user_question("gst return deadline")


def test_unrelated_subjects_do_not_inherit_the_general_gst_rate():
    evidence = [{"text": "The applicable GST rate on taxable supplies and taxable imports in Bhutan shall be 5%."}]
    assert not gst_question_has_evidence("GST rate on moon rocks", evidence)
    assert not gst_question_has_evidence("what is the weather in Thimphu", evidence)
    assert gst_question_has_evidence("GST rate", evidence)
    assert gst_question_has_evidence("GST rate on taxable imports", evidence)


def test_practical_questions_return_a_plain_answer_and_indexed_quote():
    registration = answer(SearchRequest(question="how to register for gst", target_date=date(2026, 1, 1)))
    assert "apply online" in registration["answer"]
    assert registration["evidence_quote"].startswith("Applications for registration shall be made electronically")
    assert registration["citations"][0]["document_id"] == "gst_rules_2026"

    question = "Is imported rice exempt from GST in Bhutan?"
    january = answer(SearchRequest(question=question, target_date=date(2026, 1, 1)))
    assert "depends on the type of rice" in january["answer"]
    assert "1006.30.00" in january["evidence_quote"]
    assert january["citations"][0]["document_id"] == "gst_amendment_2025"

    july = answer(SearchRequest(question=question, target_date=date(2026, 7, 1)))
    assert "brown rice" in july["answer"]
    assert july["citations"][0]["document_id"] == "gst_implementation_notice_2026"

    short = answer(SearchRequest(question="tax on rice", target_date=date(2026, 1, 1)))
    assert short["answer"].startswith("If you mean GST")
    assert "you don’t pay GST on it" in short["answer"]
    assert short["citations"][0]["chunk_id"] == january["citations"][0]["chunk_id"]


def test_answer_reports_version_history_and_what_changed():
    result = answer(SearchRequest(question="What is the GST rate?", target_date=date(2026, 1, 1)))
    assert "1 January 2026" in result["answer"]
    ctx = result["version_context"]
    assert ctx["selected"]["status"] == "in_force"
    assert [row["status"] for row in ctx["timeline"]] == ["superseded", "in_force"]
    changes = [(op["op"], op["text"]) for op in ctx["compare"]["diff"] if op["op"] != "same"]
    assert ("del", "7%.") in changes and ("add", "5%.") in changes
    assert ctx["set_aside"][0]["document_id"] == "gst_act_2020"
    assert any(item["cited"] for item in result["evidence"])


def test_chat_defaults_to_today_and_reads_before_questions_from_the_wording():
    from app.backend.main import _today

    current = answer(SearchRequest(question="What is the GST rate?"))
    assert current["target_date"] == _today() and "time_note" not in current
    assert "5%" in current["answer"]

    # Offline (no Gemini) the wording "before the latest change" still steps
    # back to the day before the current version took effect.
    before = answer(SearchRequest(question="What was the GST rate before the latest change?"))
    assert before["target_date"] == "2025-12-31"
    assert before["time_note"].startswith("Answering for 31 December 2025")
    assert "7%" in before["answer"]


def test_exemption_question_with_everyday_product_name_still_gets_evidence():
    result = answer(SearchRequest(question="Is cooking oil exempt from GST?", target_date=date(2026, 7, 1)))
    assert result["evidence"], "schedule uses trade names; the narrow filter must not empty the results"
