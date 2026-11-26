from gst_rag.version_view import VersionIndex, follow_up_dates


def chunk(chunk_id, document_id, document_type, text, start, end="", section="40", family="gst-act:40"):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_type": document_type,
        "title": document_id,
        "section_number": section,
        "page_number": 1,
        "text": text,
        "source_url": "",
        "version_family": family,
        "version_start": start,
        "version_end": end,
        "affected_sections": [],
    }


RULE = ("A taxable person may amend a return to increase a net amount within five years after the end of the "
        "tax period, but the amendment does not prevent the application of late payment penalty, "
        "administrative penalties or GST {extra}penalties.")
OLD = chunk("old", "gst_act_2020", "act", "40. " + RULE.format(extra="or EET "), "2020-01-16")
NEW = chunk("new", "gst_consolidated_2025", "consolidation",
            "40. " + RULE.format(extra="") + " [Amended by section 12 of the GST (Amendment) Act of Bhutan 2025]",
            "2026-01-01")
NOISE = chunk("noise", "gst_rules_2026", "rules", "Applications for registration shall be made electronically.", "2026-01-01",
              section="23", family="rules:23")


def test_context_marks_versions_and_diffs_the_change():
    index = VersionIndex([OLD, NEW, NOISE])
    ctx = index.context(NEW, "2026-02-01")
    assert [(row["chunk_id"], row["status"]) for row in ctx["timeline"]] == [("old", "superseded"), ("new", "in_force")]
    # The older version had no end date recorded; the next version closes it.
    assert ctx["timeline"][0]["end"] == "2025-12-31"
    changes = [(op["op"], op["text"]) for op in ctx["compare"]["diff"] if op["op"] != "same"]
    assert changes == [("del", "or EET")]
    assert ctx["compare"]["amended_by"].startswith("section 12 of the GST (Amendment) Act")
    assert [row["chunk_id"] for row in ctx["set_aside"]] == ["old"]
    assert ctx["set_aside"][0]["reason"] == "Replaced after 2025-12-31"


def test_historical_date_shows_the_later_version_as_not_yet_in_force():
    ctx = VersionIndex([OLD, NEW]).context(OLD, "2025-06-01")
    assert ctx["compare"]["direction"] == "later"
    assert ctx["set_aside"][0]["status"] == "not_yet"
    labels = follow_up_dates(ctx)
    assert labels == [{"kind": "date", "target_date": "2026-01-01", "label": "Ask about Consolidated Act (2026-01-01)"}]


def test_comparison_must_cover_the_quoted_wording():
    from gst_rag.version_view import compare_covers_quote

    ctx = VersionIndex([OLD, NEW]).context(NEW, "2026-02-01")
    # The changed clause itself is quoted: show the amendment.
    assert compare_covers_quote(ctx, "administrative penalties or GST penalties.")
    # Wording identical in both versions: the change is elsewhere in the passage.
    assert not compare_covers_quote(ctx, "A taxable person may amend a return to increase a net amount within five years")
    # Unrelated quote.
    assert not compare_covers_quote(ctx, "Applications for registration shall be made electronically to the Department")
    assert compare_covers_quote(ctx, "")


def test_unrelated_single_version_has_no_comparison():
    ctx = VersionIndex([OLD, NEW, NOISE]).context(NOISE, "2026-02-01")
    assert len(ctx["timeline"]) == 1 and ctx["compare"] is None and ctx["set_aside"] == []
