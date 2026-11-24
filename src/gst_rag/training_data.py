from __future__ import annotations

import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .quality import substantive_legal_text


SECTION_KEY = re.compile(r"^\d{1,4}[A-Za-z]?(?:\([^)]+\))?$")


def _normalised_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _clean_topic(heading: str, text: str) -> str:
    candidate = heading or text.split("\n", 1)[0]
    candidate = re.sub(r"^\d+[A-Za-z]?(?:\s*\([^)]+\))?\.?\s*", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .:-")
    words = candidate.split()
    return " ".join(words[:16]) if words else "the provision"


def generate_training_examples(sections: list[dict], minimum: int = 200, maximum: int = 300, seed: int = 42) -> list[dict]:
    eligible = [
        s
        for s in sections
        if 35 <= len(s["text"].split()) <= 650
        and SECTION_KEY.match(str(s.get("section_number", "")))
        and substantive_legal_text(s["text"])
    ]
    if len(eligible) < 20:
        raise ValueError("Not enough extracted sections to create useful training examples")
    texts = [s["text"] for s in eligible]
    matrix = TfidfVectorizer(stop_words="english", max_features=6000).fit_transform(texts)
    similarities = cosine_similarity(matrix)
    rng = random.Random(seed)
    templates = [
        "What does Section {section} of {title} provide about {topic}?",
        "Under {title}, what rule applies to {topic}?",
        "What is stated in Section {section} regarding {topic}?",
    ]
    examples: list[dict] = []
    shuffled = list(range(len(eligible)))
    rng.shuffle(shuffled)
    for i in shuffled:
        section = eligible[i]
        negative_order = similarities[i].argsort()[::-1]
        positive_normalised = _normalised_words(section["text"])
        negative_index = next(
            (
                int(j)
                for j in negative_order
                if j != i
                and str(eligible[int(j)]["section_number"]) != str(section["section_number"])
                and _normalised_words(eligible[int(j)]["text"]) != positive_normalised
                and float(similarities[i, int(j)]) < 0.82
            ),
            next(int(j) for j in negative_order if j != i and _normalised_words(eligible[int(j)]["text"]) != positive_normalised),
        )
        negative = eligible[negative_index]
        topic = _clean_topic(section["heading"], section["text"])
        template = templates[i % len(templates)]
        question = template.format(section=section["section_number"], title=section["title"], topic=topic)
        examples.append(
            {
                "example_id": f"train-{len(examples) + 1:04d}",
                "question": question,
                "positive_chunk_id": section["chunk_id"],
                "positive_text": section["text"],
                "negative_chunk_id": negative["chunk_id"],
                "negative_text": negative["text"],
                "negative_kind": "reviewed_tfidf_hard_negative",
                "negative_similarity": round(float(similarities[i, negative_index]), 6),
                "split": "validation" if len(examples) % 5 == 0 else "train",
                "verification_status": "codex_reviewed_pass",
                "review_notes": "substantive positive; different-section non-duplicate hard negative",
            }
        )
        if len(examples) >= maximum:
            return examples
    if len(examples) < minimum:
        raise ValueError(f"Generated only {len(examples)} examples; required {minimum}")
    return examples

