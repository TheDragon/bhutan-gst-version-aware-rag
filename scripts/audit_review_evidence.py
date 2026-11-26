"""Prepare auditable review sheets for the saved experiment without changing it."""

import csv
import json
import re
from collections import Counter

from pypdf import PdfReader

from gst_rag.generation import EvidenceGenerator
from gst_rag.paths import BENCHMARK, METADATA, PROCESSED, RAW_PDFS, RESULTS, TRAINING
from gst_rag.utils import read_jsonl


OUT = RESULTS / "content_review"
SYSTEMS = ("standard", "date_filtered", "version_aware")

# Corrections from reading the operative clauses. Page numbers refer to the PDF.
AMENDMENT_NOTES = {
    1: "2022 Act clause 3 amends section 3. Commencement of the amending Act (8 June 2022) is separate from the enforcement condition for Chapters 7–10.",
    2: "2022 Act clause 4 amends section 5. Repeals named in the replacement text depend on parliamentary approval for enforcement.",
    7: "Record section 11(1)(b), rather than all of section 11(1).",
    13: "Clause 18 changes the title only. Clause 19 separately changes the provision.",
    14: "Record inserted section 231A. Section 231 is the insertion anchor.",
    15: "Clause 21 changes the title only. Clause 22 separately changes the provision.",
    19: "Clause 26 inserts sections 237A and 237B after 237. Clauses 27–31 have additional insertions at the same anchor and need separate records.",
    20: "Clause 32 changes the title only. Clause 33 separately changes the provision.",
    27: "Clause 40 changes the title only.",
    40: "Record section 337(24)(b), rather than the full subsection.",
    43: "Record section 337(38)(b). Clause 57 separately changes paragraph (c).",
    44: "Record the inserted section 337(38)(d). Section 337(38)(c) is the insertion anchor.",
    46: "Record section 337(44)(b), rather than repealing the entire subsection.",
    50: "Record section 337(108)(b), rather than repealing the entire subsection.",
    52: "Clause 71 expressly targets section 3 of the 2021 Amendment Act. Preserve that instrument identity. The following table should not be treated as replacement wording without checking the surrounding schedule layout.",
    55: "Schedule X, section 2 paragraphs (1), (2) and (3) are all amended.",
    58: "Schedule X, section 21 paragraphs (1), (2) and (3) are all repealed.",
    62: "2026 Act clause 3 identifies Schedule IV C. The cited PDF page stops before the replacement material. Cite the continuation for answers about the substance of the amendment.",
}

# Scores are content judgments for inspected rows. Order is standard, date-filtered, version-aware.
ANSWER_NOTES = {
    1: ((0, 0, 0), ("General GST rate does not answer which return is used to claim the refund.",) * 3),
    2: ((0, 0, 0), ("General GST rate does not answer when the affected person may seek review.",) * 3),
    3: ((0, 0, 1), ("Invoice provision does not answer the cooling-off case.",) * 2 + ("Gives the two deadlines but the question and source chunk start mid-provision.",)),
    4: ((0, 0, 2), ("Offence-penalty definition is unrelated to registration processing.",) * 2 + ("States timely application, missed processing deadline and duration of the exception.",)),
    5: ((0, 0, 0), ("Commencement date does not state Ministry powers.",) * 2 + ("Rule 11 on group registration is a different provision from Act section 11.",)),
    6: ((1, 1, 1), ("Establishes the gift allowance generally but omits threshold and conditions.",) * 2 + ("Includes the threshold and first restriction but cuts off the personal-use condition and does not complete the rule.",)),
    7: ((2, 2, 2), ("States the outstanding period and its start and end points.",) * 3),
    8: ((1, 1, 1), ("Gives the penalty and apportionment principle but stops before the factors.",) * 3),
    9: ((1, 1, 1), ("Gives rule 14 and two control alternatives but ends with an unfinished alternative.",) * 3),
    10: ((0, 0, 0), ("General GST rate does not answer how group-member import tax is attributed.",) * 3),
    11: ((2, 2, 2), ("States all four priorities for the auction proceeds.",) * 3),
    12: ((1, 2, 0), ("Identifies the Department but adds EET from the earlier wording.", "Identifies the Department and its responsibility under the amended wording.", "Registration duration under Rule 9 answers a different question.")),
}

BENCHMARK_NOTES = {
    1: "Say Rule 234 and ask how an eligible person claims the refund. The refund-return and tax-return cases are both required.",
    2: "Ask when a person may apply to the Review Board. Cite both grounds and use the complete passage rather than an arbitrary character limit.",
    3: "The label 83(1) comes from a cross-reference in a Rules continuation. Join the prior passage and identify the actual rule before use.",
    4: "Say Rule 27 and ask when the definition exception applies. Keep both conditions and the ending condition.",
    5: "Specify section 11 of the GST Act and which Ministry powers are being asked about.",
    6: "Specify Rule 49, gifts and the conditions of the allowance. Join the next-page continuation.",
    7: "Say Rule 230 and ask for the period during which recalculated interest accrues.",
    8: "The reference answer stops before the apportionment factors. Join the continuation or narrow the question to the penalty.",
    9: "The reference answer ends mid-list. Join the next page and include the remaining grouping alternative.",
    10: "Identify the parent rule for the group-member import provision. The extracted locator contains a page-break artifact.",
    11: "Specify Rule 102 and ask for the order of priority for auction proceeds. The citation contains all four priorities.",
    12: "Specify Act section 9. Separate it from Rule 9 and keep the amended wording that refers to GST.",
}

# Whether the cited passage establishes the requested rule for the target date.
# Citation credit is based on the cited passage and target date.
CITATION_JUDGMENTS = {
    1: ("fail", "fail", "fail"), 2: ("fail", "fail", "fail"),
    3: ("fail", "fail", "partial"), 4: ("fail", "fail", "pass"),
    5: ("fail", "fail", "fail"), 6: ("partial", "partial", "partial"),
    7: ("pass", "pass", "pass"), 8: ("partial", "partial", "partial"),
    9: ("partial", "partial", "partial"), 10: ("fail", "fail", "fail"),
    11: ("pass", "pass", "pass"), 12: ("fail", "pass", "fail"),
}

OFF_TOPIC_RATE_NOTES = {
    18: "A general rate does not explain the effect on GST payable in the requested rule.",
    21: "A general rate does not describe the application of sections 327 and 328 to members.",
    28: "A general rate does not describe the historical Taxation Review provision.",
    34: "A general rate does not explain the historical scope of the exception to sections 297 and 298.",
    **{n: "The response gives a rate. The question asks when the named amendment came into force. A date embedded in a generic rate answer does not establish commencement." for n in range(76, 96)},
}

for n in range(76, 96):
    BENCHMARK_NOTES[n] = (
        "Ask when the named Amendment Act itself commenced and use a direct date reference. "
        "These questions reuse the same commencement clauses, so group them when assigning benchmark splits. "
        "If asking about a particular section, also cite the operative clause and check for distinct enforcement conditions."
    )

BENCHMARK_NOTES.update({
    46: "State the affected four chapters and preserve the parliamentary approval condition.",
    47: "Distinguish the 2022 Act commencement from when the listed repeals take effect.",
    48: "Ask which body has overall GST responsibility. Use clause 7 of the 2025 Act.",
    49: "Ask who administers GST. Use clause 8 and distinguish Act section 9 from Rule 9.",
    50: "Ask which bodies the Department assists and what the assistance concerns.",
    51: "Ask about the Department's assessment, collection and enforcement powers.",
    52: "Narrow the question to section 11(1)(b).",
    53: "The answer needs the five-year limit and the condition about penalties.",
    54: "Include both self-assessment and filing the return.",
    55: "Ask what item is listed under section 213(4).",
    56: "Ask which body now hears reviews or join the source with the grounds for review.",
    57: "Include the time limit and what triggers it.",
    58: "Ask about the title change in clause 18 or add clause 19 to cover the provision change.",
    59: "Section 231A was inserted after 231. Correct the premise.",
    60: "Ask about the title change in clause 21 or add clause 22 to cover the provision change.",
    61: "Reword as whether section 234 was repealed. This duplicates the source and fact in q-096.",
    62: "Reword as whether section 235 was repealed. This duplicates the source and fact in q-097.",
    63: "Include both the grounds limitation and the Review Board's power to grant leave.",
    64: "Clause 26 inserts 237A and 237B after 237. Specify which insertion is asked about.",
    65: "Ask about the title change in clause 32 or add clause 33 to cover the provision change.",
    66: "Ask what liabilities are third in priority under the amended subsection.",
    67: "Preserve the exception to continuation of recovery proceedings.",
    68: "State who may make the order and the standard for making it.",
    69: "Include the amount on which the penalty is charged and the restriction on compounding it.",
    70: "Ask how GST-exclusive advertising, display and quoting are treated by the amended subsection.",
    71: "Reword as whether subsection 264(6) was repealed. This duplicates q-098.",
    72: "The cited clause changes the heading. Ask for the new heading.",
    73: "Keep the distinct false representations covered by the subsection.",
    74: "To ask about the whole section, include the related subsection change in clause 43.",
    75: "Ask what the catch-all ground under section 267(3) covers.",
})

REPEAL_QUESTION_FIXES = {
    103: "Narrow to paragraph 337(44)(b).",
    106: "Narrow to paragraph 337(108)(b).",
    108: "Specify section 4 of Schedule IX.",
    109: "Specify section 1(2) of Schedule X.",
    110: "Specify section 19 of Schedule X.",
}
for n in range(96, 111):
    BENCHMARK_NOTES[n] = REPEAL_QUESTION_FIXES.get(n, "Include the 2025 Amendment Act and its clause number in the expected answer as well as the repeal itself.")
    if n in REPEAL_QUESTION_FIXES:
        continue
    if n in (96, 97):
        ANSWER_NOTES[n] = ((0, 0, 0), ("The cited material does not establish the requested repeal.",) * 2 + ("The source is a Rules provision with the same number, rather than the Act's repeal clause.",))
        CITATION_JUDGMENTS[n] = ("fail", "fail", "fail")
    else:
        ANSWER_NOTES[n] = ((0, 0, 1), ("The response and citation concern a different provision or general commencement.",) * 2 + ("The response confirms the repeal and the citation identifies the amending Act. The answer text omits the name of the amending Act.",))
        CITATION_JUDGMENTS[n] = ("fail", "fail", "pass")

TRIPLET_NOTES = {
    51: "Keep the negative. Section 124 describes scope while section 129 gives the specific intermediary treatment. Rewrite the cut-off question.",
    73: "Replace or exclude. The positive starts mid-provision and the negative supplies similar supplier/customer timing.",
    106: "Replace the negative. It reproduces part of the re-import valuation rule asked about.",
    173: "Rewrite with the Schedule X locator and specify the repeal, or exclude the item.",
    266: "The negative gives general commencement rather than the repealed provisions. Rewrite the anchor to ask which provisions are repealed.",
    297: "Retain only if the question explicitly distinguishes Act section 94 from the Rules. The negative discusses similar adjustment timing.",
    209: "Exclude as written. The positive starts with a page header and the negative contains the corresponding non-resident supply rule.",
    174: "The negative gives surrounding definitions. Rewrite the question to ask for the electronic distribution platform definition.",
    189: "Keep the negative if the question distinguishes rule 11 from rule 10. Rewrite the cut-off anchor.",
    109: "Exclude as written. The question names only the Act and the negative addresses the same mortgagee topic.",
    177: "Replace or exclude. The anchor is generic and the negative explicitly gives the provision excerpted in the positive.",
    172: "Rewrite to ask which chapters the commencement clause lists. The negative addresses a separate repeal provision.",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact(value):
    return re.sub(r"\s+", " ", value).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = {c["chunk_id"]: c for c in read_jsonl(PROCESSED / "sections.jsonl")}
    events = read_csv(METADATA / "amendment_events.csv")
    readers = {
        document: PdfReader(RAW_PDFS / f"{document}.pdf")
        for document in {e["source_document_id"] for e in events}
    }
    pages = {}
    amendment_rows = []
    for e in events:
        key = (e["source_document_id"], int(e["source_page"]))
        if key not in pages:
            pages[key] = compact(readers[key[0]].pages[key[1] - 1].extract_text())
        excerpt_matches = compact(e["evidence_excerpt"]) in pages[key]
        note = AMENDMENT_NOTES.get(int(e["event_id"].split("-")[1]))
        amendment_rows.append({
            **e, "pdf_excerpt_match": excerpt_matches,
            "content_review_status": "flagged" if note or not excerpt_matches else "locator_and_relation_supported",
            "content_review_note": note or "The operative clause names this locator and relationship.",
        })
    write_csv("amendments.csv", amendment_rows)

    benchmark = {q["question_id"]: q for q in read_jsonl(BENCHMARK / "gst_120_benchmark.jsonl")}
    gold_counts = Counter(q["gold_chunk_ids"][0] for q in benchmark.values())
    write_csv("benchmark.csv", [{
        "question_id": qid, "question": compact(q["question"]), "target_date": q["target_date"],
        "gold_answer": q["gold_answer"], "source_chunk": q["gold_chunk_ids"][0],
        "source_text": compact(chunks[q["gold_chunk_ids"][0]]["text"]),
        "questions_with_same_primary_gold": gold_counts[q["gold_chunk_ids"][0]],
        "content_review_note": BENCHMARK_NOTES.get(int(qid[2:]), ""),
    } for qid, q in benchmark.items()])

    triplets = read_jsonl(TRAINING / "minilm_training.jsonl")
    write_csv("triplets.csv", [{
        **t, "content_review_note": TRIPLET_NOTES.get(int(t["example_id"].split("-")[1]), ""),
    } for t in triplets])

    generator = EvidenceGenerator()
    answers = []
    for r in read_csv(RESULTS / "controlled_comparison_predictions.csv"):
        if r["generator"] != generator.model_name:
            raise ValueError("Citation recovery only supports the saved extractive experiment")
        evidence = [chunks[chunk_id] for chunk_id in json.loads(r["retrieved_chunk_ids"])]
        replay = generator.answer(r["question"], evidence, r["target_date"])
        matched = replay["answer"] == r["generated_answer"]
        citations = replay["citations"] if matched else []
        item_number = int(r["question_id"][2:])
        review = ANSWER_NOTES.get(item_number) if matched else None
        position = SYSTEMS.index(r["system"])
        off_topic = (
            OFF_TOPIC_RATE_NOTES.get(item_number)
            if matched and r["generated_answer"].startswith("The GST rate applicable")
            else None
        )
        score = review[0][position] if review else 0 if off_topic else ""
        note = review[1][position] if review else off_topic or ""
        citation_review = CITATION_JUDGMENTS[item_number][position] if review else "fail" if off_topic else ""
        if item_number in REPEAL_QUESTION_FIXES:
            note = "Benchmark wording needs correction before scoring. " + REPEAL_QUESTION_FIXES[item_number]
        answers.append({
            "system": r["system"], "question_id": r["question_id"], "split": r["split"],
            "question": r["question"], "answer": r["generated_answer"],
            "answer_reproduced": matched,
            "citation_chunk_ids": " | ".join(c["chunk_id"] for c in citations),
            "citation_text": " | ".join(compact(chunks[c["chunk_id"]]["text"]) for c in citations),
            "answer_score_0_2": score,
            "citation_review": citation_review,
            "content_review_note": note,
        })
    write_csv("answers.csv", answers)
    print(f"PDF excerpt matches: {sum(r['pdf_excerpt_match'] for r in amendment_rows)}/{len(events)}")
    print(f"Amendment records flagged: {sum(r['content_review_status'] == 'flagged' for r in amendment_rows)}")
    print(f"Review notes: {len(BENCHMARK_NOTES)}/{len(benchmark)} benchmark, {len(TRIPLET_NOTES)}/{len(triplets)} triplets")
    print(f"Answers reproduced: {sum(r['answer_reproduced'] for r in answers)}/{len(answers)}")
    print(f"Answers scored: {sum(r['answer_score_0_2'] != '' for r in answers)}/{len(answers)}")


if __name__ == "__main__":
    main()
