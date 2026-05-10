from __future__ import annotations

import argparse
from pathlib import Path

from backend.config import RAW_DATA_DIR
from backend.preprocessing import load_race_csv, split_race_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a RACE CSV into 80/10/10 train/val/test files.")
    parser.add_argument("--input", required=True, help="Path to the full RACE CSV file.")
    parser.add_argument("--output-dir", default=str(RAW_DATA_DIR), help="Directory for train.csv, val.csv, test.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--no-stratify", action="store_true", help="Do not preserve answer-label distribution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_race_csv(args.input)
    train_df, val_df, test_df = split_race_dataframe(
        df,
        random_state=args.seed,
        stratify_by_answer=not args.no_stratify,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Saved {len(train_df)} rows to {output_dir / 'train.csv'}")
    print(f"Saved {len(val_df)} rows to {output_dir / 'val.csv'}")
    print(f"Saved {len(test_df)} rows to {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
