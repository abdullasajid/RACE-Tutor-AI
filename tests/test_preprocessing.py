import pandas as pd

from backend.preprocessing import (
    build_answer_verification_dataset,
    split_article_sentences,
    split_race_dataframe,
)


def test_build_answer_verification_dataset(sample_race_df):
    texts, labels = build_answer_verification_dataset(sample_race_df)

    assert len(texts) == 4
    assert labels == [0, 1, 0, 0]


def test_build_answer_verification_dataset_with_hard_negative_sampling(sample_race_df):
    texts, labels = build_answer_verification_dataset(
        sample_race_df,
        negative_sampling="hard",
        negatives_per_question=1,
    )

    assert len(texts) == 2
    assert labels == [1, 0]


def test_split_article_sentences():
    sentences = split_article_sentences("First sentence. Second sentence? Third sentence!")

    assert sentences == ["First sentence.", "Second sentence?", "Third sentence!"]


def test_split_race_dataframe_uses_80_10_10_ratio():
    rows = []
    answers = ["A", "B", "C", "D"]
    for index in range(100):
        rows.append(
            {
                "id": f"sample-{index}",
                "article": "Article text.",
                "question": "Question?",
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D",
                "answer": answers[index % len(answers)],
            }
        )
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = split_race_dataframe(df, random_state=7)

    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
