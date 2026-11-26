from functools import lru_cache

from gst_rag.paths import METADATA, PROCESSED
from gst_rag.provision_story import ProvisionStories, describe_change
from gst_rag.utils import read_jsonl

TODAY = "2026-11-27"


@lru_cache(maxsize=1)
def stories() -> ProvisionStories:
    return ProvisionStories(read_jsonl(PROCESSED / "sections.jsonl"), METADATA / "amendment_events.csv")


def test_section_40_is_the_provision_not_every_passage_numbered_40():
    story = stories().story("gst_act_2020", "40", TODAY)
    assert story["status"] == "amended"
    assert story["summary"] == "Amended once · last changed on 1 January 2026 by the 2025 Amendment (clause 12)."
    newest, original = story["history"]
    assert original["kind"] == "original" and original["page"] == 40  # not the schedule rows on page 316
    assert newest["change"] == "Removed “or EET”"
    assert newest["clauses"][0]["clause"] == "12" and "is amended as" in newest["clauses"][0]["text"]
    assert story["current"]["short_name"] == "Consolidated Act" and "GST penalties" in story["current"]["text"]
    # Schedule rows, Rule 40 and the 2025 Amendment's own clause 40 are kept apart.
    assert {m["short_name"] for m in story["other_matches"]} >= {"GST Act 2020", "GST Rules 2026", "2025 Amendment"}


def test_review_board_replaces_the_tribunal():
    story = stories().story("gst_act_2020", "229", TODAY)
    assert "Tribunal" in story["history"][0]["change"] and "Board at the Ministry" in story["history"][0]["change"]


def test_repealed_section_has_no_current_text():
    story = stories().story("gst_act_2020", "234", TODAY)
    assert story["status"] == "repealed" and story["current"] is None
    assert story["summary"].startswith("Repealed on 1 January 2026 by the 2025 Amendment (clause 23)")


def test_schedule_events_do_not_leak_into_act_sections():
    # "Repealed 23" in the amendment records is Schedule X paragraph 23, not Act Section 23.
    assert stories().story("gst_act_2020", "23", TODAY)["status"] == "unchanged"
    schedule = stories().story("gst_act_2020", "Schedule X", TODAY)
    assert schedule["status"] == "amended"  # parts repealed, not the whole schedule
    assert len(schedule["history"][0]["clauses"]) == 8


def test_partial_changes_are_described_from_the_amending_clauses():
    story = stories().story("gst_act_2020", "264", TODAY)
    change = story["history"][0]["change"]
    assert "264(4) now reads" in change and "prices exclusive of GST" in change and "264(6) was repealed" in change

    definitions = stories().story("gst_act_2020", "337", TODAY)
    assert definitions["history"][0]["change"].startswith("15 parts changed")
    assert definitions["history"][0]["diff"] == []


def test_missing_original_text_falls_back_to_the_consolidated_act():
    # The 2020 Act's Section 11 wasn't extracted as its own passage; the page-302
    # paragraph that also starts "11." belongs to a schedule and must not be used.
    story = stories().story("gst_act_2020", "11", TODAY)
    assert story["title"].startswith("For the efficient implementation of this Act")
    has_earlier = any(node["short_name"] == "GST Act 2020" for node in story["history"])
    assert story["original_missing"] is (not has_earlier)


def test_exemption_schedule_links_the_notice_that_lists_new_items():
    story = stories().story("gst_act_2020", "Schedule IV C", TODAY)
    notices = story["history"][0]["notices"]
    assert notices and "Fresh butter" in notices[0]["text"]


def test_rules_and_unchanged_provisions():
    rule = stories().story("gst_rules_2026", "40", TODAY)
    assert rule["status"] == "unchanged" and rule["label"] == "Rule 40"
    assert rule["summary"] == "Unchanged since 1 January 2026."


def test_index_groups_changes_by_amending_act():
    groups = stories().index()
    assert [g["short_name"] for g in groups] == ["2026 Amendment", "2025 Amendment", "2022 Amendment"]
    labels = [item["label"] for item in groups[1]["items"]]
    assert "Section 40" in labels and labels.count("Schedule X (transitional provisions)") == 1
    # One entry per provision, even when it had several kinds of change.
    assert labels.count("Section 337") == 1 and labels.count("Section 264") == 1
    relations = {item["label"]: item["relation"] for item in groups[1]["items"]}
    assert relations["Section 234"] == "repealed" and relations["Section 264"] == "amended"


def test_describe_change_ignores_punctuation_only_edits():
    ops = [{"op": "same", "text": "a"}, {"op": "del", "text": "rate”"}, {"op": "add", "text": "rate ”"},
           {"op": "same", "text": "shall be"}, {"op": "del", "text": "7%."}, {"op": "add", "text": "5%."}]
    assert describe_change(ops) == "Replaced “7%.” with “5%.”"
