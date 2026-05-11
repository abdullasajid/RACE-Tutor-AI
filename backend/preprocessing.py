import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.config import OPTION_LABELS, REQUIRED_COLUMNS


def clean_text(text: object) -> str:
    """Normalize text for classical ML features."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_race_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def flatten_race_split(split: Any) -> pd.DataFrame:
    """Flatten a HuggingFace RACE split into article/question/A/B/C/D/answer rows."""
    rows: list[dict[str, object]] = []

    for index, example in enumerate(split):
        options = example.get("options") or []
        if len(options) != len(OPTION_LABELS):
            continue

        row = {
            "id": example.get("example_id") or example.get("id") or str(index),
            "article": example.get("article", ""),
            "question": example.get("question", ""),
            "answer": example.get("answer", ""),
        }
        row.update({label: options[position] for position, label in enumerate(OPTION_LABELS)})
        rows.append(row)

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_huggingface_race(
    dataset_name: str = "ehovy/race",
    config_name: str = "all",
) -> dict[str, pd.DataFrame]:
    """Load ehovy/race from HuggingFace and flatten train/validation/test splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the datasets package to load HuggingFace RACE: "
            "pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, config_name)
    split_names = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    return {
        output_name: flatten_race_split(dataset[input_name])
        for input_name, output_name in split_names.items()
    }


def split_race_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    stratify_by_answer: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a RACE dataframe into train/validation/test sets."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if round(total_ratio, 6) != 1.0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    stratify = df["answer"] if stratify_by_answer and "answer" in df.columns else None

    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=stratify,
        )
        temp_stratify = temp_df["answer"] if stratify is not None else None
        val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_fraction_of_temp,
            random_state=random_state,
            stratify=temp_stratify,
        )
    except ValueError:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=None,
        )
        val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_fraction_of_temp,
            random_state=random_state,
            stratify=None,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_race_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in ["article", "question", *OPTION_LABELS]:
        df[column] = df[column].fillna("").map(clean_text)
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()
    df = df[df["answer"].isin(OPTION_LABELS)]
    df = df[(df["article"] != "") & (df["question"] != "")]
    return df[REQUIRED_COLUMNS].reset_index(drop=True)


def build_option_example(article: object, question: object, option: object) -> str:
    """Create a structured text sample for one article-question-option tuple."""
    return (
        f"__ARTICLE__ {clean_text(article)} "
        f"__QUESTION__ {clean_text(question)} "
        f"__OPTION__ {clean_text(option)}"
    )


def _token_set(text: object) -> set[str]:
    return set(clean_text(text).split())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _hard_negative_score(article: str, question: str, correct_answer: str, option: str) -> float:
    option_tokens = _token_set(option)
    return (
        2.0 * _jaccard(option_tokens, _token_set(correct_answer))
        + 1.0 * _jaccard(option_tokens, _token_set(question))
        + 0.5 * _jaccard(option_tokens, _token_set(article))
    )


def build_answer_verification_dataset(
    df: pd.DataFrame,
    negative_sampling: Literal["all", "hard"] = "all",
    negatives_per_question: int = 3,
) -> tuple[list[str], list[int]]:
    """Expand each RACE row into four option-level examples."""
    if negative_sampling not in {"all", "hard"}:
        raise ValueError("negative_sampling must be 'all' or 'hard'")
    if not 1 <= negatives_per_question <= 3:
        raise ValueError("negatives_per_question must be between 1 and 3")

    prepared = prepare_race_dataframe(df)
    texts: list[str] = []
    labels: list[int] = []

    for row in prepared.itertuples(index=False):
        row_dict = row._asdict()
        article = row_dict["article"]
        question = row_dict["question"]
        correct = row_dict["answer"]

        correct_answer = row_dict.get(correct, "")
        wrong_labels = [label for label in OPTION_LABELS if label != correct]
        if negative_sampling == "hard":
            wrong_labels = sorted(
                wrong_labels,
                key=lambda label: _hard_negative_score(article, question, correct_answer, row_dict[label]),
                reverse=True,
            )[:negatives_per_question]

        selected_labels = OPTION_LABELS if negative_sampling == "all" else [correct, *wrong_labels]
        for option_label in selected_labels:
            option_text = row_dict[option_label]
            texts.append(build_option_example(article, question, option_text))
            labels.append(1 if option_label == correct else 0)

    return texts, labels


def build_option_a_binary_dataset(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """Build one binary example per question: is option A the correct answer?"""
    prepared = prepare_race_dataframe(df)
    texts: list[str] = []
    labels: list[int] = []

    for row in prepared.itertuples(index=False):
        texts.append(build_option_example(row.article, row.question, row.A))
        labels.append(1 if row.answer == "A" else 0)

    return texts, labels


def split_article_sentences(article: str) -> list[str]:
    article = "" if pd.isna(article) else str(article).strip()
    sentences = re.split(r"(?<=[.!?])\s+", article)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
