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
    weak_generated_answer,
)


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
        "section_number": item["section_number"],
        "page_number": item["page_number"],
        "applicable_from": item.get("version_start") or item.get("effective_from") or "",
        "applicable_to": item.get("version_end") or item.get("effective_to") or "",
        "source_url": item["source_url"],
    }


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
    for item in evidence:
        text = re.sub(r"\s+", " ", str(item.get("text", "")))
        match = re.search(r"applicable GST rate.*?shall be\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
        if match:
            rate_item = item
            rate = float(match.group(1))
            break
    if rate_item is None or rate is None:
        return None

    amount = _amount_from_question(question)
    vehicle = bool(re.search(r"\b(car|cars|vehicle|vehicles|automobile|motorcar)\b", question, re.I))
    subject = "an ordinary taxable passenger-car purchase or import" if vehicle else "a taxable supply or import"
    date_phrase = f" on {target_date}" if target_date else ""
    answer = f"The GST rate applicable to {subject}{date_phrase} is {rate:g}%, unless a specific exemption or zero rate applies."
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
    else:
        answer += f" Calculate it as taxable value × {rate:g}%. Ask the seller whether the quoted price already includes GST."

    calculation_item = next(
        (item for item in evidence if "computed by applying" in str(item.get("text", "")).lower()),
        None,
    )
    citations = [_citation(rate_item)]
    if calculation_item and calculation_item.get("chunk_id") != rate_item.get("chunk_id"):
        citations.append(_citation(calculation_item))
    return {
        "answer": answer,
        "abstained": False,
        "reason": "",
        "answer_strategy": "grounded_rate_calculation",
        "citations": citations,
    }


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
        grounded_rate = _grounded_rate_answer(question, evidence, target_date)
        if grounded_rate is not None:
            return {**grounded_rate, "generator": self.model_name}
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

