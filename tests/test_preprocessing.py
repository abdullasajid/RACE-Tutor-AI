import pandas as pd

from backend.feature_engineering import DENSE_FEATURE_NAMES, ModelAFeatureTransformer
from backend.preprocessing import (
    build_answer_verification_dataset,
    build_option_a_binary_dataset,
    flatten_race_split,
    prepare_race_dataframe,
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


def test_flatten_race_split_keeps_only_four_option_examples():
    split = [
        {
            "example_id": "ok",
            "article": "Article",
            "question": "Question?",
            "options": ["A1", "B1", "C1", "D1"],
            "answer": "A",
        },
        {
            "example_id": "bad",
            "article": "Article",
            "question": "Question?",
            "options": ["A1", "B1", "C1"],
            "answer": "B",
        },
    ]

    df = flatten_race_split(split)

    assert list(df["id"]) == ["ok"]
    assert df.loc[0, "A"] == "A1"
    assert df.loc[0, "D"] == "D1"


def test_prepare_race_dataframe_cleans_and_filters_invalid_rows(sample_race_df):
    df = pd.concat(
        [
            sample_race_df,
            pd.DataFrame(
                [
                    {
                        "id": "bad-answer",
                        "article": "Text",
                        "question": "Question?",
                        "A": "A",
                        "B": "B",
                        "C": "C",
                        "D": "D",
                        "answer": "E",
                    },
                    {
                        "id": "empty-article",
                        "article": "",
                        "question": "Question?",
                        "A": "A",
                        "B": "B",
                        "C": "C",
                        "D": "D",
                        "answer": "A",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    prepared = prepare_race_dataframe(df)

    assert list(prepared["id"]) == ["sample-1"]
    assert prepared.loc[0, "article"] == "ali studied hard he passed the exam"


def test_option_a_dataset_and_transformer_fit_only_train_vocabulary():
    train_df = pd.DataFrame(
        [
            {
                "id": "train-1",
                "article": "Alpha shared clue.",
                "question": "Who shared clue?",
                "A": "Alpha",
                "B": "Beta",
                "C": "Gamma",
                "D": "Delta",
                "answer": "A",
            },
            {
                "id": "train-2",
                "article": "Beta solved puzzle.",
                "question": "Who solved puzzle?",
                "A": "Alpha",
                "B": "Beta",
                "C": "Gamma",
                "D": "Delta",
                "answer": "B",
            },
        ]
    )
    val_df = pd.DataFrame(
        [
            {
                "id": "val-1",
                "article": "Zeta found treasure.",
                "question": "Who found treasure?",
                "A": "Zeta",
                "B": "Beta",
                "C": "Gamma",
                "D": "Delta",
                "answer": "A",
            }
        ]
    )

    train_texts, train_labels = build_option_a_binary_dataset(train_df)
    val_texts, val_labels = build_option_a_binary_dataset(val_df)
    transformer = ModelAFeatureTransformer(max_features=50000, min_df=1, batch_size=1)

    train_features = transformer.fit_transform(train_texts, train_labels)
    val_features = transformer.transform(val_texts)

    assert train_labels == [1, 0]
    assert val_labels == [1]
    assert "zeta" not in transformer.vectorizer_.vocabulary_
    assert train_features.shape[1] == len(transformer.vectorizer_.vocabulary_) + len(DENSE_FEATURE_NAMES)
    assert val_features.shape[1] == train_features.shape[1]
