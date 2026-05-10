from __future__ import annotations

import argparse

from backend.config import MODEL_A_DIR
from backend.model_a import (
    build_logistic_regression_model,
    build_feature_logistic_model,
    build_feature_svm_model,
    build_svm_model,
    build_tfidf_logistic_model,
    evaluate_answer_selection,
    evaluate_binary_classifier,
    save_model,
    train_answer_verifier,
)
from backend.preprocessing import build_answer_verification_dataset, load_race_csv


MODEL_BUILDERS = {
    "logreg_count": build_logistic_regression_model,
    "svm_count": build_svm_model,
    "logreg_tfidf": build_tfidf_logistic_model,
    "logreg_features": build_feature_logistic_model,
    "svm_features": build_feature_svm_model,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model A answer verifier.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--val", required=True, help="Path to val.csv")
    parser.add_argument("--model", choices=MODEL_BUILDERS, default="logreg_count")
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum vocabulary size.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = load_race_csv(args.train)
    val_df = load_race_csv(args.val)

    if args.max_rows:
        train_df = train_df.head(args.max_rows)
        val_df = val_df.head(max(1, args.max_rows // 5))

    x_train, y_train = build_answer_verification_dataset(
        train_df,
        negative_sampling=args.negative_sampling,
        negatives_per_question=args.negatives_per_question,
    )
    x_val, y_val = build_answer_verification_dataset(val_df)

    class_weight = None if args.class_weight == "none" else args.class_weight
    model = MODEL_BUILDERS[args.model](
        max_features=args.max_features,
        c=args.c,
        class_weight=class_weight,
    )
    model = train_answer_verifier(model, x_train, y_train)
    binary_metrics = evaluate_binary_classifier(model, x_val, y_val)
    answer_metrics = evaluate_answer_selection(model, val_df)

    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        output_name = args.model
        if args.negative_sampling != "all":
            output_name = f"{output_name}_{args.negative_sampling}{args.negatives_per_question}"
        if args.max_features != 20000 or args.c != 1.0 or args.class_weight != "balanced":
            output_name = f"{output_name}_mf{args.max_features}_c{args.c:g}_{args.class_weight}"

    output_path = MODEL_A_DIR / f"{output_name}.pkl"
    save_model(model, str(output_path))

    print(f"Saved model to {output_path}")
    print("Binary option-level metrics:")
    for metric, value in binary_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print("Four-option answer-selection metrics:")
    for metric, value in answer_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
