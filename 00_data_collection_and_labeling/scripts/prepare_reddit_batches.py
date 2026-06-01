"""
Prepare pre-split Reddit batch CSV files for faster labeling runs.

This script writes files like:
  batch_0000000000_0000000049.csv

Each batch file includes a __row_index column so label_reddit_data.py can
restore original row indices for prompt/label consistency.
"""

import argparse
import json
import os

import pandas as pd


def load_dataframe(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-split Reddit input into batch CSV files")
    parser.add_argument("--input", default="data/reddit.csv", help="Input CSV/XLSX path")
    parser.add_argument("--out-dir", default="data/reddit_batches", help="Output folder for batch CSV files")
    parser.add_argument("--batch-size", type=int, default=50, help="Rows per batch")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing batch_*.csv files first")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.overwrite:
        for name in os.listdir(args.out_dir):
            if name.startswith("batch_") and name.endswith(".csv"):
                os.remove(os.path.join(args.out_dir, name))

    print(f"Loading {args.input}...")
    df = load_dataframe(args.input)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    batch_count = 0
    for batch_start in range(0, total_rows, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_rows)
        batch_df = df.iloc[batch_start:batch_end].copy()
        batch_df.insert(0, "__row_index", batch_df.index)

        file_name = f"batch_{batch_start:010d}_{batch_end - 1:010d}.csv"
        file_path = os.path.join(args.out_dir, file_name)
        batch_df.to_csv(file_path, index=False)
        batch_count += 1

        if batch_count % 500 == 0:
            print(f"Prepared {batch_count} batches...")

    manifest = {
        "input": args.input,
        "out_dir": args.out_dir,
        "total_rows": total_rows,
        "batch_size": args.batch_size,
        "batch_count": batch_count,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Created {batch_count} batch files in {args.out_dir}")


if __name__ == "__main__":
    main()