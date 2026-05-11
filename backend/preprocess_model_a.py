from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import save_npz

from backend.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from backend.feature_engineering import ModelAFeatureTransformer
from backend.preprocessing import build_option_a_binary_dataset, load_race_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess RACE splits for Model A.")
    parser.add_argument("--train", default=str(RAW_DATA_DIR / "train.csv"), help="Path to train.csv.")
    parser.add_argument("--val", default=str(RAW_DATA_DIR / "val.csv"), help="Path to val.csv.")
    parser.add_argument("--test", default=str(RAW_DATA_DIR / "test.csv"), help="Path to test.csv.")
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_DATA_DIR / "model_a"),
        help="Directory where processed Model A artifacts are saved.",
    )
    parser.add_argument("--max-features", type=int, default=50000, help="Maximum vocabulary size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for cosine features.")
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=1, help="Maximum n-gram size.")
    parser.add_argument("--group-size", type=int, default=4, help="Candidate group size for per-question features.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs.")
    return parser.parse_args()


def _load_examples(path: str, max_rows: int | None = None) -> tuple[list[str], np.ndarray]:
    df = load_race_csv(path)
    if max_rows:
        df = df.head(max_rows)
    texts, labels = build_option_a_binary_dataset(df)
    return texts, np.asarray(labels, dtype=np.int8)


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_texts, y_train = _load_examples(args.train, args.max_rows)
    val_texts, y_val = _load_examples(args.val, args.max_rows)
    test_texts, y_test = _load_examples(args.test, args.max_rows)

    transformer = ModelAFeatureTransformer(
        max_features=args.max_features,
        min_df=args.min_df,
        batch_size=args.batch_size,
        ngram_max=args.ngram_max,
        group_size=args.group_size,
    )

    x_train = transformer.fit_transform(train_texts, y_train.tolist())
    x_val = transformer.transform(val_texts)
    x_test = transformer.transform(test_texts)

    save_npz(output_dir / "X_train.npz", x_train)
    save_npz(output_dir / "X_val.npz", x_val)
    save_npz(output_dir / "X_test.npz", x_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "y_test.npy", y_test)
    joblib.dump(transformer, output_dir / "feature_transformer.pkl")

    metadata = {
        "task": "Model A option-A binary classification",
        "label_definition": "1 if answer == 'A', else 0",
        "features": {
            "bow": {
                "type": "CountVectorizer",
                "binary": True,
                "max_features": args.max_features,
                "min_df": args.min_df,
                "ngram_range": [1, args.ngram_max],
                "vocabulary_size": len(transformer.vectorizer_.vocabulary_),
                "input": "article + question + option A",
            },
            "dense": transformer.dense_feature_names_,
            "dense_scaled_with": "StandardScaler fitted on train only",
        },
        "splits": {
            "train": {
                "source": args.train,
                "rows": int(x_train.shape[0]),
                "features": int(x_train.shape[1]),
                "label_counts": _label_counts(y_train),
            },
            "val": {
                "source": args.val,
                "rows": int(x_val.shape[0]),
                "features": int(x_val.shape[1]),
                "label_counts": _label_counts(y_val),
            },
            "test": {
                "source": args.test,
                "rows": int(x_test.shape[0]),
                "features": int(x_test.shape[1]),
                "label_counts": _label_counts(y_test),
            },
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved processed Model A artifacts to {output_dir}")
    print(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {x_test.shape}, y_test: {y_test.shape}")


if __name__ == "__main__":
    main()
