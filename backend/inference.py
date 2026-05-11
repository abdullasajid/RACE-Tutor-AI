from __future__ import annotations

import random

import pandas as pd

from backend.config import OPTION_LABELS
from backend.model_a import load_model, predict_best_answer
from backend.model_b import generate_distractors, generate_hints


def row_to_quiz(row: pd.Series) -> dict[str, object]:
    options = {label: str(row[label]) for label in OPTION_LABELS}
    correct_label = str(row["answer"]).strip().upper()
    correct_answer = options.get(correct_label, "")

    return {
        "article": str(row["article"]),
        "question": str(row["question"]),
        "options": options,
        "correct_label": correct_label,
        "correct_answer": correct_answer,
    }


def load_random_sample(df: pd.DataFrame) -> dict[str, object]:
    row = df.iloc[random.randrange(len(df))]
    return row_to_quiz(row)


def build_quiz_from_race_row(row: pd.Series, model_path: str | None = None) -> dict[str, object]:
    quiz = row_to_quiz(row)
    distractors = generate_distractors(
        article=str(quiz["article"]),
        question=str(quiz["question"]),
        correct_answer=str(quiz["correct_answer"]),
    )
    hints = generate_hints(
        article=str(quiz["article"]),
        question=str(quiz["question"]),
        correct_answer=str(quiz["correct_answer"]),
    )

    model_prediction = None
    if model_path:
        model = load_model(model_path)
        model_prediction = predict_best_answer(
            model=model,
            article=str(quiz["article"]),
            question=str(quiz["question"]),
            options=quiz["options"],
        )

    return {
        **quiz,
        "generated_distractors": distractors,
        "hints": hints,
        "model_prediction": model_prediction,
    }
