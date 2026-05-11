from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.config import OPTION_LABELS
from backend.feature_engineering import ModelAFeatureTransformer
from backend.preprocessing import build_option_example, clean_text, split_article_sentences


ClassWeight = str | dict[int, float] | None


def _section(text: str, start: str, end: str | None = None) -> str:
    if start not in text:
        return ""
    value = text.split(start, 1)[1]
    if end and end in value:
        value = value.split(end, 1)[0]
    return value.strip()


def _tokens(text: str) -> set[str]:
    return set(clean_text(text).split())


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _split_sections(texts: list[str]) -> tuple[list[str], list[str], list[str]]:
    articles: list[str] = []
    questions: list[str] = []
    options: list[str] = []

    for text in texts:
        articles.append(_section(text, "__ARTICLE__", "__QUESTION__"))
        questions.append(_section(text, "__QUESTION__", "__OPTION__"))
        options.append(_section(text, "__OPTION__"))

    return articles, questions, options


def _rowwise_cosine(left: csr_matrix, right: csr_matrix) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


class OptionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Handcrafted lexical features for answer verification."""

    def fit(self, texts: list[str], y: list[int] | None = None) -> "OptionFeatureExtractor":
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        rows: list[list[float]] = []
        for text in texts:
            article = _section(text, "__ARTICLE__", "__QUESTION__")
            question = _section(text, "__QUESTION__", "__OPTION__")
            option = _section(text, "__OPTION__")

            article_tokens = _tokens(article)
            question_tokens = _tokens(question)
            option_tokens = _tokens(option)
            sentence_tokens = [_tokens(sentence) for sentence in split_article_sentences(article)]
            option_sentence_scores = [
                _overlap_ratio(option_tokens, sentence_token_set)
                for sentence_token_set in sentence_tokens
            ]
            question_sentence_scores = [
                _overlap_ratio(question_tokens, sentence_token_set)
                for sentence_token_set in sentence_tokens
            ]
            question_option_tokens = question_tokens | option_tokens
            question_option_sentence_scores = [
                _overlap_ratio(question_option_tokens, sentence_token_set)
                for sentence_token_set in sentence_tokens
            ]
            max_sentence_overlap = max(option_sentence_scores, default=0.0)
            max_question_sentence_overlap = max(question_sentence_scores, default=0.0)
            max_question_option_sentence_overlap = max(question_option_sentence_scores, default=0.0)
            best_question_sentence_tokens = set()
            if question_sentence_scores:
                best_question_sentence_tokens = sentence_tokens[int(np.argmax(question_sentence_scores))]

            rows.append(
                [
                    float(bool(option_tokens)),
                    float(option in article and bool(option)),
                    len(option_tokens),
                    _overlap_ratio(option_tokens, article_tokens),
                    _overlap_ratio(option_tokens, question_tokens),
                    max_sentence_overlap,
                    len(option_tokens & article_tokens),
                    len(option_tokens & question_tokens),
                    max_question_sentence_overlap,
                    max_question_option_sentence_overlap,
                    _overlap_ratio(option_tokens, best_question_sentence_tokens),
                    len(option_tokens & best_question_sentence_tokens),
                ]
            )

        return csr_matrix(np.asarray(rows, dtype=float))


class TfidfSimilarityFeatureExtractor(BaseEstimator, TransformerMixin):
    """TF-IDF cosine features between article, question, and candidate option."""

    def __init__(self, max_features: int = 20000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
        )

    def fit(self, texts: list[str], y: list[int] | None = None) -> "TfidfSimilarityFeatureExtractor":
        articles, questions, options = _split_sections(texts)
        self.vectorizer.fit([*articles, *questions, *options])
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        articles, questions, options = _split_sections(texts)
        question_options = [f"{question} {option}" for question, option in zip(questions, options)]

        article_matrix = self.vectorizer.transform(articles)
        question_matrix = self.vectorizer.transform(questions)
        option_matrix = self.vectorizer.transform(options)
        question_option_matrix = self.vectorizer.transform(question_options)

        features = np.column_stack(
            [
                _rowwise_cosine(option_matrix, article_matrix),
                _rowwise_cosine(option_matrix, question_matrix),
                _rowwise_cosine(question_matrix, article_matrix),
                _rowwise_cosine(question_option_matrix, article_matrix),
            ]
        )
        return csr_matrix(features)


class TextAndOptionFeatures(BaseEstimator, TransformerMixin):
    """Combine text, lexical, and TF-IDF similarity features."""

    def __init__(self, max_features: int = 20000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.option_features = OptionFeatureExtractor()
        self.similarity_features = TfidfSimilarityFeatureExtractor(max_features=max_features)

    def fit(self, texts: list[str], y: list[int] | None = None) -> "TextAndOptionFeatures":
        self.vectorizer.fit(texts)
        self.option_features.fit(texts, y)
        self.similarity_features.fit(texts, y)
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        text_features = self.vectorizer.transform(texts)
        option_features = self.option_features.transform(texts)
        similarity_features = self.similarity_features.transform(texts)
        return hstack([text_features, option_features, similarity_features], format="csr")


def build_logistic_regression_model(
    max_features: int = 20000,
    c: float = 1.0,
    class_weight: ClassWeight = "balanced",
) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", CountVectorizer(max_features=max_features, stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight, C=c)),
        ]
    )


def build_svm_model(
    max_features: int = 20000,
    c: float = 1.0,
    class_weight: ClassWeight = "balanced",
) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", CountVectorizer(max_features=max_features, stop_words="english")),
            ("classifier", LinearSVC(class_weight=class_weight, C=c, max_iter=3000)),
        ]
    )


def build_tfidf_logistic_model(
    max_features: int = 20000,
    c: float = 1.0,
    class_weight: ClassWeight = "balanced",
) -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight, C=c)),
        ]
    )


def build_feature_logistic_model(
    max_features: int = 20000,
    c: float = 1.0,
    class_weight: ClassWeight = "balanced",
) -> Pipeline:
    return Pipeline(
        [
            ("features", TextAndOptionFeatures(max_features=max_features)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight, C=c)),
        ]
    )


def build_feature_svm_model(
    max_features: int = 20000,
    c: float = 1.0,
    class_weight: ClassWeight = "balanced",
) -> Pipeline:
    return Pipeline(
        [
            ("features", TextAndOptionFeatures(max_features=max_features)),
            ("classifier", LinearSVC(class_weight=class_weight, C=c, max_iter=3000)),
        ]
    )


def build_option_a_logistic_model(
    max_features: int = 50000,
    min_df: int = 2,
    c: float = 1.0,
    class_weight: ClassWeight = "balanced",
    batch_size: int = 4096,
    ngram_max: int = 1,
    group_size: int = 4,
) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ModelAFeatureTransformer(
                    max_features=max_features,
                    min_df=min_df,
                    batch_size=batch_size,
                    ngram_max=ngram_max,
                    group_size=group_size,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight, C=c)),
        ]
    )


def build_option_a_svm_model(
    max_features: int = 50000,
    min_df: int = 2,
    c: float = 1.0,
    class_weight: ClassWeight = "balanced",
    batch_size: int = 4096,
    ngram_max: int = 1,
    group_size: int = 4,
) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ModelAFeatureTransformer(
                    max_features=max_features,
                    min_df=min_df,
                    batch_size=batch_size,
                    ngram_max=ngram_max,
                    group_size=group_size,
                ),
            ),
            ("classifier", LinearSVC(class_weight=class_weight, C=c, max_iter=3000)),
        ]
    )


def train_answer_verifier(model: Pipeline, texts: list[str], labels: list[int]) -> Pipeline:
    return model.fit(texts, labels)


def evaluate_binary_classifier(model: Pipeline, texts: list[str], labels: list[int]) -> dict[str, object]:
    predictions = model.predict(texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def evaluate_answer_selection(model: Pipeline, df: pd.DataFrame) -> dict[str, object]:
    """Evaluate the option-level model as a four-choice answer selector."""
    y_true: list[str] = []
    option_examples: list[str] = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        y_true.append(str(row_dict["answer"]).strip().upper())
        for label in OPTION_LABELS:
            option_examples.append(
                build_option_example(row_dict["article"], row_dict["question"], row_dict[label])
            )

    scores = _score_texts(model, option_examples).reshape(-1, len(OPTION_LABELS))
    y_pred = [OPTION_LABELS[index] for index in scores.argmax(axis=1)]

    return {
        "answer_accuracy": accuracy_score(y_true, y_pred),
        "answer_macro_f1": f1_score(y_true, y_pred, labels=OPTION_LABELS, average="macro", zero_division=0),
        "answer_confusion_matrix": confusion_matrix(y_true, y_pred, labels=OPTION_LABELS).tolist(),
    }


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def _score_texts(model: Pipeline, texts: list[str]) -> np.ndarray:
    classifier = model.named_steps.get("classifier")
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(texts))[:, 1]
    if hasattr(classifier, "decision_function"):
        return np.asarray(model.decision_function(texts), dtype=float)
    return np.asarray(model.predict(texts), dtype=float)


def _score_option(model: Pipeline, text: str) -> float:
    return float(_score_texts(model, [text])[0])


def predict_best_answer(model: Pipeline, article: str, question: str, options: dict[str, str]) -> dict[str, object]:
    article = clean_text(article)
    question = clean_text(question)
    scores = {}

    for label in OPTION_LABELS:
        combined = build_option_example(article, question, options[label])
        scores[label] = _score_option(model, combined)

    best_label = max(scores, key=scores.get)
    raw_scores = np.array(list(scores.values()), dtype=float)
    shifted = raw_scores - raw_scores.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()

    return {
        "answer": best_label,
        "scores": scores,
        "confidence": float(probabilities[OPTION_LABELS.index(best_label)]),
    }
