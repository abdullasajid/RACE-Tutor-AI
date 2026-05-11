from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from backend.config import OPTION_LABELS
from backend.preprocessing import split_article_sentences


STOPWORDS = set(ENGLISH_STOP_WORDS)
FALLBACK_DISTRACTORS = [
    "A different passage detail",
    "An unsupported conclusion",
    "A less relevant idea",
]


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z]{2,}\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _format_phrase(tokens: tuple[str, ...]) -> str:
    return " ".join(tokens).title()


def _candidate_phrases(article: str, max_length: int = 4) -> Counter[tuple[str, ...]]:
    """Extract lightweight phrase candidates without external NLP models."""
    phrase_counts: Counter[tuple[str, ...]] = Counter()

    for sentence in split_article_sentences(article):
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z]{2,}\b", sentence.lower())
        content_positions = [
            (position, token)
            for position, token in enumerate(tokens)
            if token not in STOPWORDS
        ]
        content_tokens = [token for _position, token in content_positions]

        for start in range(len(content_tokens)):
            for length in range(1, max_length + 1):
                end = start + length
                if end > len(content_tokens):
                    continue
                phrase = tuple(content_tokens[start:end])
                if not phrase:
                    continue
                phrase_counts[phrase] += 1

    return phrase_counts


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _phrase_score(
    phrase: tuple[str, ...],
    frequency: int,
    question_tokens: set[str],
    correct_tokens: set[str],
) -> float:
    phrase_tokens = set(phrase)
    question_overlap = len(phrase_tokens & question_tokens)
    correct_overlap = _jaccard(phrase_tokens, correct_tokens)
    length_bonus = 0.25 * min(len(phrase), 3)
    return frequency + question_overlap + length_bonus - (0.75 * correct_overlap)


def _is_valid_distractor(phrase: tuple[str, ...], correct_answer: str, existing: set[str]) -> bool:
    text = _format_phrase(phrase)
    normalized = _normalize(text)
    correct_normalized = _normalize(correct_answer)
    if not normalized or normalized in existing:
        return False
    if normalized == correct_normalized:
        return False
    if normalized in correct_normalized or correct_normalized in normalized:
        return False
    if len(set(phrase)) == 1 and len(phrase) > 1:
        return False
    return True


def generate_distractors(
    article: str,
    correct_answer: str,
    question: str = "",
    count: int = 3,
) -> list[str]:
    """Generate plausible extractive distractors from passage phrases."""
    correct_tokens = set(_content_tokens(correct_answer))
    question_tokens = set(_content_tokens(question))
    candidates = _candidate_phrases(article)

    ranked = sorted(
        candidates.items(),
        key=lambda item: _phrase_score(item[0], item[1], question_tokens, correct_tokens),
        reverse=True,
    )

    distractors: list[str] = []
    seen: set[str] = set()
    for phrase, _frequency in ranked:
        if not _is_valid_distractor(phrase, correct_answer, seen):
            continue
        if correct_tokens and set(phrase) == correct_tokens:
            continue
        distractor = _format_phrase(phrase)
        distractors.append(distractor)
        seen.add(_normalize(distractor))
        if len(distractors) == count:
            break

    fallback_index = 0
    while len(distractors) < count:
        fallback = FALLBACK_DISTRACTORS[fallback_index % len(FALLBACK_DISTRACTORS)]
        fallback_index += 1
        if _normalize(fallback) not in seen:
            distractors.append(fallback)
            seen.add(_normalize(fallback))

    return distractors


def _overlap_score(sentence: str, query: str) -> int:
    sentence_tokens = set(_content_tokens(sentence))
    query_tokens = set(_content_tokens(query))
    return len(sentence_tokens & query_tokens)


def _mask_answer_terms(sentence: str, correct_answer: str) -> str:
    masked = sentence
    for token in sorted(set(_content_tokens(correct_answer)), key=len, reverse=True):
        masked = re.sub(rf"\b{re.escape(token)}\b", "[...]", masked, flags=re.IGNORECASE)
    return masked


def _question_focus(question: str) -> str:
    tokens = _content_tokens(question)
    if not tokens:
        return "the key details in the question"
    return ", ".join(tokens[:4])


def generate_hints(article: str, question: str, correct_answer: str, count: int = 3) -> list[str]:
    """Return graduated hints from broad guidance to direct evidence."""
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

    best_sentence = ranked[0]
    support_sentence = _mask_answer_terms(best_sentence, correct_answer)
    selected = [
        f"Focus on the passage details connected to: {_question_focus(question)}.",
        support_sentence,
        best_sentence,
    ]

    while len(selected) < count:
        selected.insert(0, "Review the passage and eliminate options that are not supported.")

    return selected[:count]


def build_generated_options(
    article: str,
    question: str,
    correct_answer: str,
    correct_label: str = "A",
) -> dict[str, str]:
    """Create a four-option set using the correct answer plus generated distractors."""
    labels = list(OPTION_LABELS)
    if correct_label not in labels:
        correct_label = "A"

    distractors = generate_distractors(
        article=article,
        question=question,
        correct_answer=correct_answer,
        count=len(labels) - 1,
    )
    options: dict[str, str] = {}
    distractor_index = 0
    for label in labels:
        if label == correct_label:
            options[label] = correct_answer
        else:
            options[label] = distractors[distractor_index]
            distractor_index += 1
    return options
