import os

from gst_rag.gemini import grounded_answer, load_env_file, rewrite_query
from gst_rag.generation import EvidenceGenerator


class FakeClient:
    model = "fake-gemini"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def generate_json(self, system, prompt, schema):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def passage(chunk_id, text, section="1"):
    return {
        "chunk_id": chunk_id,
        "document_id": "gst_consolidated_2025",
        "title": "Consolidated GST Act",
        "section_number": section,
        "page_number": 1,
        "text": text,
        "source_url": "https://example.test/act.pdf",
        "version_start": "2026-01-01",
    }


RATE = passage("c1", "Other than the zero-rate prescribed in Schedule V, the applicable GST rate on taxable supplies and taxable imports of goods and services shall be 5%.")
NOTICE = passage("c2", "GST shall apply to all taxable supplies and imports destined for consumption within the country.", "2")


def test_grounded_answer_keeps_only_retrieved_citations_and_verbatim_quotes():
    client = FakeClient({
        "answerable": True,
        "answer": "Yes. The standard GST rate is 5%. The passages shown don't mention water specifically.",
        "supporting_ids": ["E1", "E9"],
        "quote": "the applicable GST rate on taxable supplies and taxable imports of goods and services shall be 5%.",
    })
    result, note = grounded_answer(client, "is gst taxable on water", [RATE, NOTICE], "2026-01-01")
    assert note == ""
    assert result["answer_strategy"] == "llm_grounded"
    assert [item["chunk_id"] for item in result["cited_items"]] == ["c1"]
    assert result["evidence_quote"].endswith("shall be 5%.")
    assert "[E1]" in client.prompts[0] and "2026-01-01" in client.prompts[0]


def test_grounded_answer_drops_invented_quotes_and_rejects_uncited_answers():
    invented = FakeClient({"answerable": True, "answer": "Rice is exempt.", "supporting_ids": ["E1"], "quote": "Rice is exempt from GST."})
    result, _ = grounded_answer(invented, "is rice taxed", [RATE], None)
    assert result["evidence_quote"] == ""

    uncited = FakeClient({"answerable": True, "answer": "Rice is exempt.", "supporting_ids": ["E7"], "quote": ""})
    result, note = grounded_answer(uncited, "is rice taxed", [RATE], None)
    assert result is None and "cite" in note


def test_grounded_answer_abstains_when_model_says_evidence_is_insufficient():
    client = FakeClient({"answerable": False, "answer": "", "supporting_ids": [], "quote": ""})
    result, _ = grounded_answer(client, "is rice taxed", [NOTICE], None)
    assert result["abstained"] is True
    assert result["reason"] == "llm_insufficient_evidence"


def test_generator_uses_llm_and_falls_back_offline_on_error():
    good = FakeClient({"answerable": True, "answer": "Yes, the standard rate is 5%.", "supporting_ids": ["E1"], "quote": ""})
    result = EvidenceGenerator(llm=good).answer("is gst taxable on water", [RATE, NOTICE], "2026-01-01")
    assert result["answer_strategy"] == "llm_grounded"
    assert result["generator"] == "fake-gemini"
    assert result["citations"][0]["chunk_id"] == "c1"

    broken = FakeClient(error=RuntimeError("Gemini HTTP 503"))
    result = EvidenceGenerator(llm=broken).answer("is gst taxable on water", [RATE, NOTICE], "2026-01-01")
    assert result["answer_strategy"] == "extractive"
    assert "Gemini HTTP 503" in result["llm_note"]


def test_rewrite_removes_terms_that_would_change_version_filtering():
    client = FakeClient({
        "clean_question": "Is rice subject to GST?",
        "time_intent": "current",
        "time_date": "2024-01-01",
        "search_terms": "rice exempt supply Schedule GST rate amended changed 2025",
    })
    result = rewrite_query(client, "is rcie taxed with gst", "2026-11-27")
    assert result == {
        "search_terms": "rice exempt supply Schedule GST rate",
        "clean_question": "Is rice subject to GST?",
        "time_intent": "current",
        "time_date": "",  # a date only counts when the intent is 'date'
    }
    assert "2026-11-27" in client.prompts[0]
    assert rewrite_query(FakeClient(error=TimeoutError()), "x") is None


def test_rewrite_validates_time_intent():
    dated = FakeClient({"clean_question": "What was the GST rate in 2024?", "time_intent": "date", "time_date": "2024-12-31", "search_terms": "GST rate"})
    assert rewrite_query(dated, "gst rate in 2024")["time_date"] == "2024-12-31"
    bad_date = FakeClient({"clean_question": "x y z", "time_intent": "date", "time_date": "last year", "search_terms": "GST"})
    assert rewrite_query(bad_date, "gst last year")["time_intent"] == "current"
    previous = FakeClient({"clean_question": "What was the rate before?", "time_intent": "previous", "search_terms": "GST rate"})
    assert rewrite_query(previous, "rate before the change")["time_intent"] == "previous"


def test_earlier_versions_explain_changes_but_can_never_be_cited():
    earlier = {**RATE, "chunk_id": "old", "text": RATE["text"].replace("5%", "7%"), "_version_label": "EARLIER VERSION"}
    client = FakeClient({
        "answerable": True,
        "answer": "The standard rate is 5%.",
        "supporting_ids": ["V1", "E1"],
        "quote": "",
        "change_summary": "Before 1 January 2026 the rate was 7%; it is now 5%.",
        "answer_quality": "complete",
        "better_question": "Ignored when the answer is complete",
    })
    result, _ = grounded_answer(client, "What is the GST rate?", [RATE], "2026-01-01", related=[earlier])
    assert [item["chunk_id"] for item in result["cited_items"]] == ["c1"]
    assert "[V1]" in client.prompts[0] and "EARLIER VERSION" in client.prompts[0]
    assert result["change_summary"].startswith("Before 1 January 2026")
    assert result["suggestion"] == ""

    without_versions, _ = grounded_answer(client, "What is the GST rate?", [RATE], "2026-01-01")
    assert without_versions["change_summary"] == ""


def test_weak_answers_carry_a_better_question():
    partial = FakeClient({
        "answerable": True, "answer": "The standard rate is 5%; electric vehicles aren't mentioned.",
        "supporting_ids": ["E1"], "quote": "", "answer_quality": "partial",
        "better_question": "What is the standard GST rate on taxable supplies",
    })
    result, _ = grounded_answer(partial, "gst on EVs?", [RATE], "2026-01-01")
    assert result["suggestion"] == "What is the standard GST rate on taxable supplies?"

    none = FakeClient({"answerable": False, "answer": "", "supporting_ids": [], "quote": "", "answer_quality": "none", "better_question": "When must a GST return be filed?"})
    result, _ = grounded_answer(none, "something odd", [NOTICE], None)
    assert result["abstained"] and result["suggestion"] == "When must a GST return be filed?"


def test_generator_keeps_rate_template_for_price_calculations():
    llm = FakeClient({"answerable": True, "answer": "unused", "supporting_ids": ["E1"], "quote": ""})
    result = EvidenceGenerator(llm=llm).answer("GST on a car costing Nu. 1,000,000?", [RATE], "2026-01-01")
    assert result["answer_strategy"] == "grounded_rate_calculation"
    assert "1 January 2026" in result["answer"] and "Nu. 50,000" in result["answer"]
    assert llm.prompts == []

    ev = EvidenceGenerator(llm=llm).answer("Is there GST on an electric vehicle?", [RATE], "2026-01-01")
    assert ev["answer_strategy"] == "llm_grounded"


def test_env_file_does_not_override_existing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nGST_TEST_A=from_file\nGST_TEST_B='quoted'\nGST_TEST_EMPTY=\n", encoding="utf-8")
    monkeypatch.setenv("GST_TEST_A", "from_shell")
    monkeypatch.delenv("GST_TEST_B", raising=False)
    monkeypatch.delenv("GST_TEST_EMPTY", raising=False)
    load_env_file(env)
    assert os.environ["GST_TEST_A"] == "from_shell"
    assert os.environ["GST_TEST_B"] == "quoted"
    assert "GST_TEST_EMPTY" not in os.environ
    monkeypatch.delenv("GST_TEST_B")
