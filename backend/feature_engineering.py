from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from backend.preprocessing import clean_text


NEGATION_WORDS = {
    "ain",
    "aren",
    "cannot",
    "couldn",
    "didn",
    "doesn",
    "don",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "never",
    "no",
    "none",
    "nor",
    "not",
    "nothing",
    "nowhere",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
}

DENSE_FEATURE_NAMES = [
    "article_length",
    "question_length",
    "option_length",
    "question_article_overlap",
    "option_article_overlap",
    "option_coverage_ratio",
    "article_option_cosine",
    "option_question_overlap",
    "option_question_coverage",
    "question_has_negation",
    "option_has_negation",
    "question_option_negation_mismatch",
    "question_option_article_tfidf_cosine",
    "question_article_tfidf_cosine",
    "option_article_tfidf_cosine",
    "option_length_ratio_to_group_mean",
    "option_article_overlap_delta_from_group_mean",
    "option_coverage_delta_from_group_mean",
]


def _section(text: str, start: str, end: str | None = None) -> str:
    if start not in text:
        return ""
    value = text.split(start, 1)[1]
    if end and end in value:
        value = value.split(end, 1)[0]
    return value.strip()


def _token_set(text: str) -> set[str]:
    return set(clean_text(text).split())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def split_option_examples(texts: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split structured article/question/option examples into their sections."""
    articles: list[str] = []
    questions: list[str] = []
    options: list[str] = []

    for text in texts:
        articles.append(clean_text(_section(text, "__ARTICLE__", "__QUESTION__")))
        questions.append(clean_text(_section(text, "__QUESTION__", "__OPTION__")))
        options.append(clean_text(_section(text, "__OPTION__")))

    return articles, questions, options


def combine_article_question_option(
    articles: list[str],
    questions: list[str],
    options: list[str],
) -> list[str]:
    """Combine article + question + candidate option for bag-of-words encoding."""
    return [
        clean_text(f"{article} {question} {option}")
        for article, question, option in zip(articles, questions, options)
    ]


def build_lexical_features(
    articles: list[str],
    questions: list[str],
    options: list[str],
) -> np.ndarray:
    """Build handcrafted lexical features used by Model A."""
    rows: list[list[float]] = []

    for article, question, option in zip(articles, questions, options):
        article_tokens = _token_set(article)
        question_tokens = _token_set(question)
        option_tokens = _token_set(option)
        option_coverage = (
            len(option_tokens & article_tokens) / len(option_tokens)
            if option_tokens
            else 0.0
        )

        rows.append(
            [
                float(len(clean_text(article).split())),
                float(len(clean_text(question).split())),
                float(len(clean_text(option).split())),
                _jaccard(question_tokens, article_tokens),
                _jaccard(option_tokens, article_tokens),
                option_coverage,
                _jaccard(option_tokens, question_tokens),
                len(option_tokens & question_tokens) / len(option_tokens)
                if option_tokens
                else 0.0,
                float(bool(question_tokens & NEGATION_WORDS)),
                float(bool(option_tokens & NEGATION_WORDS)),
                float(bool(question_tokens & NEGATION_WORDS) != bool(option_tokens & NEGATION_WORDS)),
            ]
        )

    return np.asarray(rows, dtype=float)


def compute_article_option_cosine(
    articles: list[str],
    options: list[str],
    vectorizer: CountVectorizer,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute cosine similarity between article and option vectors in batches."""
    values = np.zeros(len(articles), dtype=float)

    for start in range(0, len(articles), batch_size):
        end = start + batch_size
        article_matrix = vectorizer.transform(articles[start:end])
        option_matrix = vectorizer.transform(options[start:end])

        dot_product = np.asarray(article_matrix.multiply(option_matrix).sum(axis=1)).ravel()
        article_norm = np.sqrt(np.asarray(article_matrix.multiply(article_matrix).sum(axis=1)).ravel())
        option_norm = np.sqrt(np.asarray(option_matrix.multiply(option_matrix).sum(axis=1)).ravel())
        denominator = article_norm * option_norm
        values[start:end] = np.divide(
            dot_product,
            denominator,
            out=np.zeros_like(dot_product, dtype=float),
            where=denominator > 0,
        )

    return values.reshape(-1, 1)


def compute_pairwise_cosine(
    left_texts: list[str],
    right_texts: list[str],
    vectorizer: CountVectorizer | TfidfVectorizer,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute cosine similarity between two aligned text lists in batches."""
    values = np.zeros(len(left_texts), dtype=float)

    for start in range(0, len(left_texts), batch_size):
        end = start + batch_size
        left_matrix = vectorizer.transform(left_texts[start:end])
        right_matrix = vectorizer.transform(right_texts[start:end])

        dot_product = np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()
        left_norm = np.sqrt(np.asarray(left_matrix.multiply(left_matrix).sum(axis=1)).ravel())
        right_norm = np.sqrt(np.asarray(right_matrix.multiply(right_matrix).sum(axis=1)).ravel())
        denominator = left_norm * right_norm
        values[start:end] = np.divide(
            dot_product,
            denominator,
            out=np.zeros_like(dot_product, dtype=float),
            where=denominator > 0,
        )

    return values.reshape(-1, 1)


def build_group_relative_features(
    articles: list[str],
    questions: list[str],
    options: list[str],
    lexical_features: np.ndarray,
    group_size: int = 4,
) -> np.ndarray:
    """Build option features normalized against sibling options for a question."""
    rows = np.zeros((len(options), 3), dtype=float)
    if group_size <= 1:
        return rows

    for start in range(0, len(options), group_size):
        end = start + group_size
        if end > len(options):
            break
        if len(set(articles[start:end])) != 1 or len(set(questions[start:end])) != 1:
            continue

        option_lengths = lexical_features[start:end, 2]
        option_article_overlap = lexical_features[start:end, 4]
        option_coverage = lexical_features[start:end, 5]
        mean_option_length = option_lengths.mean()

        rows[start:end, 0] = np.divide(
            option_lengths,
            mean_option_length,
            out=np.ones_like(option_lengths, dtype=float),
            where=mean_option_length > 0,
        )
        rows[start:end, 1] = option_article_overlap - option_article_overlap.mean()
        rows[start:end, 2] = option_coverage - option_coverage.mean()

    return rows


def build_dense_features(
    articles: list[str],
    questions: list[str],
    options: list[str],
    vectorizer: CountVectorizer,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    batch_size: int = 4096,
    group_size: int = 4,
) -> np.ndarray:
    lexical_features = build_lexical_features(articles, questions, options)
    cosine_feature = compute_article_option_cosine(
        articles,
        options,
        vectorizer=vectorizer,
        batch_size=batch_size,
    )
    dense_blocks = [lexical_features[:, :6], cosine_feature, lexical_features[:, 6:]]

    if tfidf_vectorizer is None:
        tfidf_features = np.zeros((len(options), 3), dtype=float)
    else:
        question_options = [
            clean_text(f"{question} {option}")
            for question, option in zip(questions, options)
        ]
        tfidf_features = np.column_stack(
            [
                compute_pairwise_cosine(
                    question_options,
                    articles,
                    vectorizer=tfidf_vectorizer,
                    batch_size=batch_size,
                ),
                compute_pairwise_cosine(
                    questions,
                    articles,
                    vectorizer=tfidf_vectorizer,
                    batch_size=batch_size,
                ),
                compute_pairwise_cosine(
                    options,
                    articles,
                    vectorizer=tfidf_vectorizer,
                    batch_size=batch_size,
                ),
            ]
        )

    dense_blocks.append(tfidf_features)
    dense_blocks.append(
        build_group_relative_features(
            articles,
            questions,
            options,
            lexical_features=lexical_features,
            group_size=group_size,
        )
    )
    return np.column_stack(dense_blocks)


class ModelAFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Binary bag-of-words + scaled dense features for Model A.

    The CountVectorizer and StandardScaler are fitted only in ``fit`` on the
    training examples, then reused unchanged for validation/test transforms.
    """

    def __init__(
        self,
        max_features: int = 50000,
        min_df: int = 2,
        batch_size: int = 4096,
        ngram_max: int = 1,
        group_size: int = 4,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.batch_size = batch_size
        self.ngram_max = ngram_max
        self.group_size = group_size

    def fit(self, texts: list[str], y: list[int] | None = None) -> "ModelAFeatureTransformer":
        articles, questions, options = split_option_examples(texts)
        combined_texts = combine_article_question_option(articles, questions, options)

        self.vectorizer_ = CountVectorizer(
            binary=True,
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=(1, self.ngram_max),
        )
        self.vectorizer_.fit(combined_texts)
        self.tfidf_vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=(1, self.ngram_max),
            sublinear_tf=True,
        )
        self.tfidf_vectorizer_.fit(combined_texts)

        dense_features = build_dense_features(
            articles,
            questions,
            options,
            vectorizer=self.vectorizer_,
            tfidf_vectorizer=self.tfidf_vectorizer_,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )
        self.scaler_ = StandardScaler()
        self.scaler_.fit(dense_features)
        self.dense_feature_names_ = DENSE_FEATURE_NAMES
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        check_is_fitted(self, ["vectorizer_", "tfidf_vectorizer_", "scaler_"])
        articles, questions, options = split_option_examples(texts)
        combined_texts = combine_article_question_option(articles, questions, options)

        bow_features = self.vectorizer_.transform(combined_texts)
        dense_features = build_dense_features(
            articles,
            questions,
            options,
            vectorizer=self.vectorizer_,
            tfidf_vectorizer=self.tfidf_vectorizer_,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )
        scaled_dense_features = self.scaler_.transform(dense_features)
        return hstack([bow_features, csr_matrix(scaled_dense_features)], format="csr")
