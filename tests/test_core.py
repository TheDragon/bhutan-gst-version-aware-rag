from gst_rag.corpus import _page_chunks
from gst_rag.generation import EvidenceGenerator
from gst_rag.quality import (
    act_commencement_clause,
    best_answer_evidence,
    commencement_intent,
    evidence_supports_question,
    expand_retrieval_query,
    gst_rate_intent,
    rate_passage_adjustment,
    substantive_adjustment,
    toc_like,
    weak_generated_answer,
)
from gst_rag.retrieval import parse_query_date, reciprocal_rank_fusion, valid_on
from gst_rag.versioning import normalize_section, version_valid_on


def test_query_date_parsing():
    assert parse_query_date("What applied on 2026-01-05?") == "2026-01-05"
    assert parse_query_date("What applied in 2025?") == "2025-12-31"
    assert parse_query_date("What is current?") is None


def test_valid_period_filter():
    record = {"effective_from": "2025-01-01", "effective_to": "2025-12-31"}
    assert valid_on(record, "2025-06-01")
    assert not valid_on(record, "2026-01-01")
    assert valid_on(record, None)


def test_rrf_prefers_shared_result():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
    assert fused[0][0] in {"a", "b"}
    assert {item[0] for item in fused} == {"a", "b", "c", "d"}


def test_section_parser():
    chunks = _page_chunks("1. This Act is the Goods and Services Tax Act of Bhutan 2020.\nIt extends to the whole of Bhutan and contains additional legal wording for testing.\n\n2. This provision commences on a specified date and contains enough words for a useful retrievable passage in the corpus.")
    assert len(chunks) == 2
    assert chunks[0][0] == "1"


def test_version_metadata_helpers():
    assert normalize_section("Section 40 (2)") == "40(2)"
    record = {"version_start": "2026-01-01", "version_end": "2026-05-17"}
    assert version_valid_on(record, "2026-03-01")
    assert not version_valid_on(record, "2026-05-18")


def test_contents_chunks_are_demoted():
    contents = "12. Section 40 of the GST Act is amended as:........................4"
    operative = (
        "Section 40 is amended as: A taxable person may amend a return within five years "
        "after the end of the tax period, but penalties may still apply."
    )
    assert toc_like(contents)
    assert not toc_like(operative)
    assert substantive_adjustment({"text": operative}) > substantive_adjustment({"text": contents})


def test_one_word_generation_falls_back_to_substantive_evidence():
    contents = {"text": "Section 40 is amended as:........................4"}
    operative = {
        "text": "A taxable person may amend a return within five years, but applicable penalties remain."
    }
    assert weak_generated_answer("CHANGE")
    assert not weak_generated_answer("A taxable person may amend the return within five years.")
    assert best_answer_evidence([contents, operative]) is operative


def test_exact_section_evidence_can_safely_fallback():
    evidence = {
        "section_number": "12",
        "affected_sections": ["40"],
        "text": (
            "Section 40 is amended as: A taxable person may amend a return within five years "
            "after the tax period, but penalties still apply."
        ),
    }
    assert evidence_supports_question("How was Section 40 amended?", evidence)
    assert not evidence_supports_question(
        "How was Section 70 amended?",
        evidence,
    )


def test_vehicle_rate_question_expands_to_schedule():
    question = "What GST is implied on a car?"
    expanded = expand_retrieval_query(question)
    assert gst_rate_intent(question)
    assert "Schedule I" in expanded
    rate_item = {"text": "The applicable GST rate on taxable supplies shall be 5%."}
    formula_item = {"text": "Apply the GST rate prescribed under Schedule I."}
    assert rate_passage_adjustment(question, rate_item) > rate_passage_adjustment(question, formula_item)


def test_grounded_vehicle_rate_calculation():
    shared = {
        "document_id": "gst_consolidated_2025",
        "title": "Consolidated GST Act",
        "source_url": "https://example.test/gst.pdf",
        "version_start": "2026-01-01",
        "version_end": "",
        "effective_from": "2026-01-01",
        "effective_to": "",
    }
    evidence = [
        {
            **shared,
            "chunk_id": "rate",
            "section_number": "1",
            "page_number": 146,
            "text": "The applicable GST rate on taxable supplies and taxable imports shall be 5%.",
        },
        {
            **shared,
            "chunk_id": "formula",
            "section_number": "25",
            "page_number": 20,
            "text": "GST payable shall be computed by applying the GST rate to the value of the supply or import.",
        },
    ]
    result = EvidenceGenerator().answer(
        "What GST applies to a car costing Nu. 1,000,000 before GST?",
        evidence,
        "2026-01-01",
    )
    assert result["answer_strategy"] == "grounded_rate_calculation"
    assert "5%" in result["answer"]
    assert "Nu. 50,000" in result["answer"]
    assert "Nu. 1,050,000" in result["answer"]


def test_named_act_commencement_answer_uses_the_cited_clause():
    question = "What commencement date applies to section 40 under the GST Amendment Act 2025?"
    evidence = {
        "chunk_id": "commencement",
        "document_id": "gst_amendment_2025",
        "document_type": "amendment",
        "title": "GST Amendment Act 2025",
        "source_url": "https://example.test/amendment.pdf",
        "section_number": "2",
        "page_number": 18,
        "text": "2. This Act comes into force on 1 January 2026. Savings",
        "named_act_commencement": True,
    }

    assert commencement_intent(question)
    assert not gst_rate_intent(question)
    assert act_commencement_clause({**evidence, "text": "2. Commencement..............1"}) == ""
    result = EvidenceGenerator().answer(question, [evidence], "2025-12-31")
    assert result["answer_strategy"] == "act_commencement_clause"
    assert "1 January 2026" in result["answer"]
    assert "2025-12-31" not in result["answer"]
    assert result["citations"][0]["chunk_id"] == "commencement"

