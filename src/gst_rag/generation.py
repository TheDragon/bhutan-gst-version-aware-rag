from __future__ import annotations

import os
import re
from dataclasses import dataclass

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from .quality import (
    act_commencement_clause,
    best_answer_evidence,
    bm25_tokenize,
    commencement_intent,
    evidence_supports_question,
    gst_rate_intent,
    tobacco_context,
    weak_generated_answer,
)
from .versioning import version_valid_on


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def _extractive_answer(question: str, evidence: dict, sentence_limit: int = 2) -> str:
    query_terms = set(bm25_tokenize(question)) - {
        "what", "which", "does", "the", "and", "for", "from", "under", "about", "section"
    }
    candidates = _sentences(evidence.get("text", ""))
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (-len(query_terms & set(bm25_tokenize(pair[1]))), pair[0]),
    )
    chosen = sorted(ranked[:sentence_limit], key=lambda pair: pair[0])
    return " ".join(sentence for _, sentence in chosen)[:900]


def _citation(item: dict) -> dict:
    return {
        "chunk_id": item["chunk_id"],
        "document_id": item["document_id"],
        "title": item["title"],
        "document_type": item.get("document_type", ""),
        "affected_sections": list(item.get("affected_sections", [])),
        "section_number": item["section_number"],
        "page_number": item["page_number"],
        "applicable_from": item.get("version_start") or item.get("effective_from") or "",
        "applicable_to": item.get("version_end") or item.get("effective_to") or "",
        "source_url": item["source_url"],
    }


def present_answer(question: str, result: dict, evidence: list[dict]) -> dict:
    """Put readable wording and a source excerpt in separate fields for Ask & Inspect."""
    if result["abstained"] or result.get("evidence_quote") or not result.get("citations"):
        return result
    source_id = result["citations"][0]["chunk_id"]
    source = next((item for item in evidence if item["chunk_id"] == source_id), None)
    if source is None:
        return result

    answer = result["answer"]
    if result.get("answer_strategy") in {"extractive", "validated_extractive_fallback"}:
        answer = re.sub(r"^\d{1,4}[a-z]?\.\s+", "", answer, flags=re.I)
        amendment = re.match(r"Section ([\dA-Za-z()]+)\b[^:]{0,240}\bis amended as:\s*[“\"]?", answer, re.I)
        if amendment:
            answer = f"Here’s what the amendment says about Section {amendment[1]}. " + answer[amendment.end():].strip('“”" ')
        for pattern, replacement in (
            (r"\bthe GST on a taxable supply of imported B2B services must be payable by the customer who must be treated as incurring both output tax and input tax, but subject to this Act, the customer must be allowed an input tax credit for that supply provided t\s*he customer has accounted for the output tax\.",
             "If your business buys a taxable service from an overseas business, your business is responsible for reporting and paying the GST. You can claim a credit for it if you report the GST and meet the conditions in the Act."),
            (r"\bB2B services\b", "services one business buys from another"),
            (r"\bA taxable person shall,\s*no later than thirty days after the end of each tax period,\s*file a GST return\b",
             "If you’re a taxable person, you must file your GST return within 30 days after each tax period ends"),
            (r"\bA taxable person may\b", "If you’re a taxable person, you can"),
            (r"\breflecting the person’s net amount for the period\b", "and show your net GST for that period"),
            (r"\bthe amendment does not prevent the application of ([^.]+)\.", r"\1 can still apply."),
            (r"\bshall not be applicable\b", "does not apply"),
            (r"\bshall not\b", "must not"),
            (r"\bshall\b", "must"),
            (r"\bprior to\b", "before"),
            (r"\bin respect of\b", "for"),
            (r"\bsubsequent to\b", "after"),
        ):
            answer = re.sub(pattern, replacement, answer, flags=re.I)
    elif result.get("answer_strategy") == "grounded_rate_calculation":
        answer = re.sub(
            r"^The GST rate applicable to (.*?) is (\d+(?:\.\d+)?%), unless a specific exemption or zero rate applies\.",
            r"The standard GST rate is \2 for \1. Some transactions qualify for no GST.",
            answer,
        )

    text = re.sub(r"^\s*\d{1,4}[a-z]?\.\s+", "", source["text"], flags=re.I)
    excerpt = _extractive_answer(question, {**source, "text": text}, sentence_limit=1)
    return {**result, "answer": answer, "evidence_quote": excerpt if len(excerpt) >= 15 else ""}


def lookup_practical_gst_answer(
    question: str, chunks: list[dict], target_date: str | None, top_k: int
) -> tuple[dict, list[dict]] | None:
    """Look up operative evidence when a general search can confuse nearby rules."""
    normalized = " ".join(bm25_tokenize(question))
    if re.fullmatch(r"how (?:do i |do we |can i |can we |to )?(?:register for|apply for) gst(?: registration)?(?: in bhutan)?", normalized):
        source = next(
            (item for item in chunks if item["document_type"] == "rules" and version_valid_on(item, target_date)
             and "Applications for registration shall be made" in item["text"]),
            None,
        )
        if source:
            quote = re.search(r"Applications for registration shall be made[^.]*\.", " ".join(source["text"].split()))
            if quote:
                return _guided_reply(
                    "To register for GST, apply online to Bhutan’s Department of Revenue and Customs or use the application method it provides. Rule 23 explains how to send your application.",
                    quote.group(), [source],
                )
    rice_tax = re.fullmatch(
        r"(?:(?:what is the )?(?:gst|tax)(?: rate)? (?:on|for) (?:imported )?rice|(?:imported )?rice (?:gst|tax))"
        r"(?: in bhutan)?(?: latest| today| as of now)?",
        normalized,
    )
    if rice_tax or (re.search(r"\brice\b", normalized) and re.search(r"\bexempt\w*\b", normalized)):
        eligible = [item for item in chunks if version_valid_on(item, target_date)]
        schedule = next((item for item in eligible if item["document_id"] == "gst_amendment_2025"
                         and "1006.30.00" in item["text"] and item["section_number"] == "70"), None)
        scope = next((item for item in eligible if item["document_type"] == "consolidation"
                      and "exempt supply or exempt import" in " ".join(item["text"].split()) and item["section_number"] == "26"), None)
        expanded = next((item for item in eligible if item["document_id"] == "gst_implementation_notice_2026"
                         and "1006.20.20" in item["text"] and "1006.30.00" in item["text"]), None)
        if schedule and scope:
            evidence = [schedule, scope][:top_k]
            message = "It depends on the type of rice. Semi-milled and fully milled rice are on the exemption list under product code 1006.30.00."
            if expanded:
                schedule = expanded
                evidence = [expanded, scope][:top_k]
                message = "It depends on the type of rice. The updated exemption list includes brown rice, red rice (Yeechum), and semi-milled or fully milled rice."
            elif target_date and target_date >= "2026-05-18":
                message += " The exemption list changed in 2026, so check the import date before relying on it for another type of rice."
            rice_entries = r"1006\.30\.00\s*-?\s*Semi-milled or wholly milled rice, whether or not polished or glazed"
            if expanded:
                rice_entries = r"1006\.20\.10.*?1006\.20\.20.*?" + rice_entries
            quote = re.search(rice_entries, " ".join(schedule["text"].split()), re.I)
            if quote:
                if top_k > 1:
                    message += " If the rice you import matches a listed product code, you don’t pay GST on it."
                if rice_tax:
                    message = "If you mean GST, here’s how it works. " + message
                return _guided_reply(message, quote.group(), evidence)
    return None


def _guided_reply(answer: str, quote: str, evidence: list[dict]) -> tuple[dict, list[dict]]:
    return {
        "answer": answer,
        "abstained": False,
        "reason": "",
        "answer_strategy": "guided_lookup",
        "evidence_quote": quote,
        "citations": [_citation(item) for item in evidence],
    }, evidence


def lookup_gst_definition(chunks: list[dict], target_date: str | None, top_k: int) -> tuple[dict, list[dict]] | None:
    definition_pattern = re.compile(r'["“]GST["”]\s+or\s+["“]Goods and Services Tax["”]\s+means\b', re.I)
    scope_pattern = re.compile(r'GST shall apply to all taxable supplies and imports destined for consumption within the country\.', re.I)
    definitions = []
    scope_notices = []
    for item in chunks:
        if not version_valid_on(item, target_date):
            continue
        text = " ".join(item["text"].split())
        if item["document_type"] in {"act", "consolidation"} and definition_pattern.search(text):
            definitions.append(item)
        if item["document_type"] == "notice" and scope_pattern.search(text):
            scope_notices.append(item)

    if not definitions:
        return None

    def rank(item: dict) -> int:
        return item.get("version_rank", 0)

    evidence = [max(definitions, key=rank)]
    answer = "GST stands for Goods and Services Tax."
    definition = definition_pattern.search(" ".join(evidence[0]["text"].split())).group()
    if top_k > 1 and scope_notices:
        evidence.append(max(scope_notices, key=rank))
        answer += " In Bhutan, it applies to taxable supplies and imports for consumption in the country."
    return {
        "answer": answer,
        "abstained": False,
        "reason": "",
        "answer_strategy": "definition_lookup",
        "evidence_quote": definition.rsplit(" ", 1)[0],
        "citations": [_citation(item) for item in evidence],
    }, evidence


def readable_date(value: str) -> str:
    """'2026-01-01' -> '1 January 2026' (unchanged if it isn't an ISO date)."""
    from datetime import date

    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{parsed.day} {parsed:%B %Y}"


def _money(value: float) -> str:
    return f"Nu. {value:,.0f}" if value.is_integer() else f"Nu. {value:,.2f}"


def _amount_from_question(question: str) -> float | None:
    match = re.search(r"\b(?:nu\.?|ngultrum)\s*([\d,]+(?:\.\d+)?)", question, re.I)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _grounded_rate_answer(question: str, evidence: list[dict], target_date: str | None) -> dict | None:
    if not gst_rate_intent(question):
        return None
    rate_item = None
    rate = None
    rate_quote = ""
    tobacco = tobacco_context(question)
    rate_pattern = r"applicable GST rate.*?shall be\s*(\d+(?:\.\d+)?)\s*%"
    if tobacco:
        rate_pattern = (
            r"(?:Other than the zero[- ]rate prescribed in Schedule V\s*,\s*)?"
            r"(?:the\s+)?applicable GST rate[^.]*?shall be\s*(\d+(?:\.\d+)?)\s*%\.?"
        )
    for item in evidence:
        text = re.sub(r"\s+", " ", str(item.get("text", "")))
        match = re.search(rate_pattern, text, re.I)
        if match:
            rate_item = item
            rate = float(match.group(1))
            rate_quote = match.group()
            break
    if rate_item is None or rate is None:
        if tobacco:
            return {
                "answer": "I couldn’t confirm the applicable GST rate from the official passages I found for the selected date.",
                "abstained": True,
                "reason": "no_supported_gst_rate",
                "answer_strategy": "abstention",
                "citations": [],
            }
        return None

    amount = _amount_from_question(question)
    vehicle = bool(re.search(r"\b(car|cars|vehicle|vehicles|automobile|motorcar)\b", question, re.I))
    subject = "an ordinary taxable passenger-car purchase or import" if vehicle else "a taxable supply or import"
    date_phrase = f" on {readable_date(target_date)}" if target_date else ""
    answer = f"The GST rate applicable to {subject}{date_phrase} is {rate:g}%, unless a specific exemption or zero rate applies."
    if tobacco:
        product = "cigarettes" if re.search(r"\bcigarettes?\b", question, re.I) else "tobacco products"
        answer = f"If you import {product} into Bhutan, the standard GST rate is {rate:g}%."
        if amount is None:
            answer += f" That’s {_money(rate)} for every Nu. 100 that GST is charged on."
        answer += " Some imports qualify for no GST. A separate tax called excise may also apply."
    if amount is not None:
        inclusive = bool(re.search(r"\b(including|inclusive|includes)\s+GST\b|\bGST[- ]inclusive\b", question, re.I))
        exclusive = bool(re.search(r"\b(before|excluding|exclusive of|plus)\s+GST\b|\bGST[- ]exclusive\b", question, re.I))
        if inclusive and not exclusive:
            tax = amount * rate / (100 + rate)
            base = amount - tax
            answer += f" For a GST-inclusive price of {_money(amount)}, the GST component is {_money(tax)} and the pre-GST value is {_money(base)}."
        else:
            tax = amount * rate / 100
            total = amount + tax
            answer += f" For a price of {_money(amount)} before GST, the GST is {_money(tax)} and the total after GST is {_money(total)}."
    elif not tobacco:
        answer += f" Calculate it as taxable value × {rate:g}%. Ask the seller whether the quoted price already includes GST."

    calculation_item = next(
        (item for item in evidence if "computed by applying" in str(item.get("text", "")).lower()),
        None,
    )
    citations = [_citation(rate_item)]
    if calculation_item and calculation_item.get("chunk_id") != rate_item.get("chunk_id"):
        citations.append(_citation(calculation_item))
    response = {
        "answer": answer,
        "abstained": False,
        "reason": "",
        "answer_strategy": "grounded_rate_calculation",
        "citations": citations,
    }
    if tobacco:
        response["evidence_quote"] = rate_quote
    return response


@dataclass
class EvidenceGenerator:
    """Evidence-bound answer generator used identically across all systems.

    The default extractive mode is deterministic and works offline.  A Hugging
    Face instruction model can be supplied for the proposal's generator-model
    comparison; citations are still assembled from metadata rather than left to
    the language model.
    """

    model_name: str = "extractive-evidence-v1"
    max_new_tokens: int = 96
    llm: object | None = None  # optional gst_rag.gemini.GeminiClient

    def __post_init__(self) -> None:
        self.pipeline = None
        if self.model_name != "extractive-evidence-v1":
            from transformers import pipeline

            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,
                framework="pt",
            )

    def answer(
        self,
        question: str,
        evidence: list[dict],
        target_date: str | None = None,
        related: list[dict] | None = None,
    ) -> dict:
        if not evidence:
            return {
                "answer": "I cannot establish the applicable GST provision from the indexed official evidence.",
                "abstained": True,
                "reason": "no_retrieved_evidence",
                "generator": self.model_name,
                "citations": [],
            }

        top = best_answer_evidence(evidence)
        if commencement_intent(question) and top.get("named_act_commencement"):
            clause = act_commencement_clause(top)
            if clause:
                answer = f"The Act's commencement clause states: {clause}"
                if re.search(r"\b(section|provision|schedule)\b", question, re.I):
                    answer += " A particular provision may have a separate commencement or enforcement condition."
                return {
                    "answer": answer,
                    "abstained": False,
                    "reason": "",
                    "generator": self.model_name,
                    "answer_strategy": "act_commencement_clause",
                    "citations": [_citation(top)],
                }
        # A price calculation needs exact arithmetic, so the rate template keeps
        # those. Every other question goes to the answer model first, so a
        # product the template doesn't know (an electric vehicle, say) isn't
        # forced into the general car-rate answer.
        prefer_template = self.llm is None or _amount_from_question(question) is not None
        if prefer_template:
            grounded_rate = _grounded_rate_answer(question, evidence, target_date)
            if grounded_rate is not None:
                return {**grounded_rate, "generator": self.model_name}
        llm_note = ""
        if self.llm is not None:
            from .gemini import grounded_answer

            llm_result, llm_note = grounded_answer(self.llm, question, evidence, target_date, related)
            if llm_result is not None:
                cited = llm_result.pop("cited_items", [])
                if cited:
                    llm_result["citations"] = [_citation(item) for item in cited]
                return {**llm_result, "generator": self.llm.model}
            if not prefer_template:
                grounded_rate = _grounded_rate_answer(question, evidence, target_date)
                if grounded_rate is not None:
                    return {**grounded_rate, "generator": self.model_name, "llm_note": f"Gemini unavailable, used offline answer: {llm_note}"}
        fallback = self._local_answer(question, evidence, top, target_date)
        if llm_note:
            fallback["llm_note"] = f"Gemini unavailable, used offline answer: {llm_note}"
        return fallback

    def _local_answer(self, question: str, evidence: list[dict], top: dict, target_date: str | None) -> dict:
        answer_strategy = "extractive" if self.pipeline is None else "instruction_model"
        if self.pipeline is None:
            answer = _extractive_answer(question, top)
        else:
            context = "\n\n".join(
                f"[{i}] {item['title']}, section {item['section_number']}, page {item['page_number']}: {item['text'][:1200]}"
                for i, item in enumerate(evidence[:3], start=1)
            )
            prompt = (
                "Answer only from the official GST evidence below. "
                "If the evidence is insufficient, answer 'INSUFFICIENT EVIDENCE'. "
                "State the actual legal rule or change in one to three complete sentences. "
                "Include any time limit, condition, and penalty effect stated in the evidence. "
                "Do not answer with a category or one-word label such as 'change' or 'amended'.\n"
                f"Question: {question}\nTarget date: {target_date or 'not stated'}\nEvidence:\n{context}\nAnswer:"
            )
            generated = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                truncation=True,
            )[0]["generated_text"].strip()
            if "INSUFFICIENT EVIDENCE" in generated.upper():
                if evidence_supports_question(question, top):
                    answer = _extractive_answer(question, top)
                    answer_strategy = "validated_extractive_fallback"
                else:
                    return {
                        "answer": "I cannot establish the applicable GST provision from the indexed official evidence.",
                        "abstained": True,
                        "reason": "generator_insufficient_evidence",
                        "generator": self.model_name,
                        "answer_strategy": "abstention",
                        "citations": [],
                    }
            elif weak_generated_answer(generated):
                answer = _extractive_answer(question, top)
                answer_strategy = "validated_extractive_fallback"
            else:
                answer = generated

        return {
            "answer": answer,
            "abstained": False,
            "reason": "",
            "generator": self.model_name,
            "answer_strategy": answer_strategy,
            "citations": [_citation(top)],
        }

