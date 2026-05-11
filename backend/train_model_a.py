from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from backend.config import MODEL_A_DIR, RAW_DATA_DIR
from backend.model_a import (
    build_logistic_regression_model,
    build_feature_logistic_model,
    build_feature_svm_model,
    build_option_a_logistic_model,
    build_option_a_svm_model,
    build_svm_model,
    build_tfidf_logistic_model,
    evaluate_answer_selection,
    evaluate_binary_classifier,
    save_model,
    train_answer_verifier,
)
from backend.preprocessing import (
    build_answer_verification_dataset,
    build_option_a_binary_dataset,
    load_huggingface_race,
    load_race_csv,
    prepare_race_dataframe,
)


MODEL_BUILDERS = {
    "logreg_option_a": build_option_a_logistic_model,
    "svm_option_a": build_option_a_svm_model,
    "logreg_candidate": build_option_a_logistic_model,
    "svm_candidate": build_option_a_svm_model,
    "logreg_count": build_logistic_regression_model,
    "svm_count": build_svm_model,
    "logreg_tfidf": build_tfidf_logistic_model,
    "logreg_features": build_feature_logistic_model,
    "svm_features": build_feature_svm_model,
}
OPTION_A_MODELS = {"logreg_option_a", "svm_option_a"}
MODEL_A_FEATURE_MODELS = {"logreg_option_a", "svm_option_a", "logreg_candidate", "svm_candidate"}
CANDIDATE_MODELS = {"logreg_candidate", "svm_candidate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model A answer verifier.")
    parser.add_argument(
        "--source",
        choices=["csv", "huggingface"],
        default="csv",
        help="Load local CSV files or ehovy/race from HuggingFace.",
    )
    parser.add_argument("--train", default=None, help="Path to train.csv when --source csv is used.")
    parser.add_argument("--val", default=None, help="Path to val.csv when --source csv is used.")
    parser.add_argument("--test", default=None, help="Optional path to test.csv when --source csv is used.")
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Optional preprocessed Model A directory with X_*.npz, y_*.npy, and feature_transformer.pkl.",
    )
    parser.add_argument("--hf-dataset", default="ehovy/race", help="HuggingFace dataset name.")
    parser.add_argument("--hf-config", default="all", help="HuggingFace RACE config: all, high, or middle.")
    parser.add_argument(
        "--save-flattened",
        action="store_true",
        help="Save flattened HuggingFace splits to data/raw.",
    )
    parser.add_argument("--model", choices=MODEL_BUILDERS, default="logreg_option_a")
    parser.add_argument("--max-features", type=int, default=50000, help="Maximum vocabulary size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for option-A BOW.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for cosine feature computation.")
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=1, help="Maximum n-gram size.")
    parser.add_argument("--group-size", type=int, default=4, help="Candidate group size for per-question features.")
    parser.add_argument("--c", type=float, default=1.0, help="Regularization strength parameter C.")
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Classifier class weighting strategy.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick experiments")
    parser.add_argument(
        "--negative-sampling",
        choices=["all", "hard"],
        default="all",
        help="Use all wrong options or only the hardest wrong options during training.",
    )
    parser.add_argument(
        "--negatives-per-question",
        type=int,
        default=3,
        help="Number of wrong options to keep when --negative-sampling hard is used.",
    )
    parser.add_argument("--output-name", default=None, help="Optional model filename stem.")
    args = parser.parse_args()
    if args.processed_dir is None and args.source == "csv" and (not args.train or not args.val):
        parser.error("--train and --val are required when --source csv is used")
    if args.processed_dir is not None and args.model not in OPTION_A_MODELS:
        parser.error("--processed-dir currently supports only option-A models")
    return args


def _load_dataframes(args: argparse.Namespace):
    if args.source == "huggingface":
        splits = load_huggingface_race(args.hf_dataset, args.hf_config)
        if args.save_flattened:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            splits["train"].to_csv(RAW_DATA_DIR / "train.csv", index=False)
            splits["val"].to_csv(RAW_DATA_DIR / "val.csv", index=False)
            splits["test"].to_csv(RAW_DATA_DIR / "test.csv", index=False)
        return splits["train"], splits["val"], splits["test"]

    train_df = load_race_csv(args.train)
    val_df = load_race_csv(args.val)
    test_df = load_race_csv(args.test) if args.test else None
    return train_df, val_df, test_df


def _build_model(args: argparse.Namespace):
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.model in MODEL_A_FEATURE_MODELS:
        return MODEL_BUILDERS[args.model](
            max_features=args.max_features,
            min_df=args.min_df,
            c=args.c,
            class_weight=class_weight,
            batch_size=args.batch_size,
            ngram_max=args.ngram_max,
            group_size=args.group_size,
        )

    return MODEL_BUILDERS[args.model](
        max_features=args.max_features,
        c=args.c,
        class_weight=class_weight,
    )


def _build_training_examples(args: argparse.Namespace, train_df, val_df, test_df):
    if args.model in OPTION_A_MODELS:
        x_train, y_train = build_option_a_binary_dataset(train_df)
        x_val, y_val = build_option_a_binary_dataset(val_df)
        test_examples = build_option_a_binary_dataset(test_df) if test_df is not None else None
        return x_train, y_train, x_val, y_val, test_examples

    if args.model in CANDIDATE_MODELS:
        x_train, y_train = build_answer_verification_dataset(train_df)
        x_val, y_val = build_answer_verification_dataset(val_df)
        test_examples = build_answer_verification_dataset(test_df) if test_df is not None else None
        return x_train, y_train, x_val, y_val, test_examples

    x_train, y_train = build_answer_verification_dataset(
        train_df,
        negative_sampling=args.negative_sampling,
        negatives_per_question=args.negatives_per_question,
    )
    x_val, y_val = build_answer_verification_dataset(val_df)
    test_examples = build_answer_verification_dataset(test_df) if test_df is not None else None
    return x_train, y_train, x_val, y_val, test_examples


def _load_processed_features(processed_dir: str):
    processed_path = Path(processed_dir)
    return {
        "x_train": load_npz(processed_path / "X_train.npz"),
        "x_val": load_npz(processed_path / "X_val.npz"),
        "x_test": load_npz(processed_path / "X_test.npz"),
        "y_train": np.load(processed_path / "y_train.npy"),
        "y_val": np.load(processed_path / "y_val.npy"),
        "y_test": np.load(processed_path / "y_test.npy"),
        "transformer": joblib.load(processed_path / "feature_transformer.pkl"),
    }


def _evaluate_feature_matrix_classifier(classifier, features, labels: np.ndarray) -> dict[str, object]:
    predictions = classifier.predict(features)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def _train_from_processed(args: argparse.Namespace) -> tuple[Pipeline, dict[str, object], dict[str, object] | None]:
    processed = _load_processed_features(args.processed_dir)
    base_model = _build_model(args)
    classifier = base_model.named_steps["classifier"]
    classifier.fit(processed["x_train"], processed["y_train"])

    model = Pipeline(
        [
            ("features", processed["transformer"]),
            ("classifier", classifier),
        ]
    )
    binary_metrics = _evaluate_feature_matrix_classifier(
        classifier,
        processed["x_val"],
        processed["y_val"],
    )
    test_binary_metrics = _evaluate_feature_matrix_classifier(
        classifier,
        processed["x_test"],
        processed["y_test"],
    )
    return model, binary_metrics, test_binary_metrics


def main() -> None:
    args = parse_args()

    answer_metrics = None
    test_binary_metrics = None
    test_answer_metrics = None

    if args.processed_dir is not None:
        model, binary_metrics, test_binary_metrics = _train_from_processed(args)
        if args.val:
            val_df = load_race_csv(args.val)
            if args.max_rows:
                val_df = val_df.head(max(1, args.max_rows // 5))
            answer_metrics = evaluate_answer_selection(model, prepare_race_dataframe(val_df))
        if args.test:
            test_df = load_race_csv(args.test)
            if args.max_rows:
                test_df = test_df.head(max(1, args.max_rows // 5))
            test_answer_metrics = evaluate_answer_selection(model, prepare_race_dataframe(test_df))
    else:
        train_df, val_df, test_df = _load_dataframes(args)

        if args.max_rows:
            train_df = train_df.head(args.max_rows)
            val_df = val_df.head(max(1, args.max_rows // 5))
            if test_df is not None:
                test_df = test_df.head(max(1, args.max_rows // 5))

        x_train, y_train, x_val, y_val, test_examples = _build_training_examples(
            args,
            train_df,
            val_df,
            test_df,
        )

        model = _build_model(args)
        model = train_answer_verifier(model, x_train, y_train)
        binary_metrics = evaluate_binary_classifier(model, x_val, y_val)
        answer_metrics = evaluate_answer_selection(model, prepare_race_dataframe(val_df))
        if test_df is not None and test_examples is not None:
            x_test, y_test = test_examples
            test_binary_metrics = evaluate_binary_classifier(model, x_test, y_test)
            test_answer_metrics = evaluate_answer_selection(model, prepare_race_dataframe(test_df))

    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        output_name = args.model
        if args.model not in MODEL_A_FEATURE_MODELS and args.negative_sampling != "all":
            output_name = f"{output_name}_{args.negative_sampling}{args.negatives_per_question}"
        if (
            args.max_features != 50000
            or args.min_df != 2
            or args.ngram_max != 1
            or args.c != 1.0
            or args.class_weight != "balanced"
        ):
            output_name = (
                f"{output_name}_mf{args.max_features}_mindf{args.min_df}"
                f"_ng{args.ngram_max}_c{args.c:g}_{args.class_weight}"
            )

    output_path = MODEL_A_DIR / f"{output_name}.pkl"
    save_model(model, str(output_path))

    print(f"Saved model to {output_path}")
    print("Binary option-level metrics:")
    for metric, value in binary_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    if answer_metrics is not None:
        print("Four-option answer-selection metrics:")
        for metric, value in answer_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    if test_binary_metrics is not None:
        print("Test binary option-A metrics:")
        for metric, value in test_binary_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    if test_answer_metrics is not None:
        print("Test four-option answer-selection metrics:")
        for metric, value in test_answer_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
