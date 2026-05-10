from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from backend.preprocessing import split_article_sentences


STOPWORDS = set(ENGLISH_STOP_WORDS)


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z]{2,}\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def generate_distractors(article: str, correct_answer: str, count: int = 3) -> list[str]:
    """Generate simple extractive distractors from frequent passage terms."""
    correct_tokens = set(_content_tokens(correct_answer))
    candidates = Counter(_content_tokens(article))

    distractors: list[str] = []
    for token, _frequency in candidates.most_common():
        if token in correct_tokens:
            continue
        if token.lower() in correct_answer.lower():
            continue
        if token not in distractors:
            distractors.append(token.title())
        if len(distractors) == count:
            break

    while len(distractors) < count:
        distractors.append(f"Option {len(distractors) + 1}")

    return distractors


def _overlap_score(sentence: str, query: str) -> int:
    sentence_tokens = set(_content_tokens(sentence))
    query_tokens = set(_content_tokens(query))
    return len(sentence_tokens & query_tokens)


def generate_hints(article: str, question: str, correct_answer: str, count: int = 3) -> list[str]:
    """Return ranked extractive hints from broad to specific."""
    sentences = split_article_sentences(article)
    if not sentences:
        return ["Read the passage carefully.", "Look for related details.", "Compare each option."]

    query = f"{question} {correct_answer}"
    ranked = sorted(
        sentences,
        key=lambda sentence: (
            _overlap_score(sentence, query),
            _overlap_score(sentence, correct_answer),
            len(sentence),
        ),
        reverse=True,
    )

    selected = ranked[:count]
    selected = list(reversed(selected))

    while len(selected) < count:
        selected.insert(0, "Review the passage and eliminate options that are not supported.")

    return selected
