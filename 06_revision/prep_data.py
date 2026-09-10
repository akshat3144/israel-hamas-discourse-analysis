# -*- coding: utf-8 -*-
"""
Stage 1 of the paper revision pipeline.

Addresses reviewer point R20 (matched observation window) by restricting BOTH
platforms to the common study window, and produces compact working files so the
downstream analysis does not have to re-read ~1 GB of CSV each time.

Temporal semantics (reviewer R9/R20):
  * Reddit  -> `created_time`  = true comment timestamp (reliable)
  * YouTube -> `video_date`    = video publish date. YouTube's `created_time` is a
                                 scrape-date artifact (spans 2024-2026) and is NOT
                                 a valid comment timeline.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENT = ROOT / "02_emotional_tone_analysis" / "outputs"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Common study window (R20): start of the war -> end of collection
WINDOW_START = pd.Timestamp("2023-10-07")
WINDOW_END = pd.Timestamp("2024-05-07")

R_COLS = ["index", "Label", "self_text", "score", "author_name", "controversiality",
          "created_time", "subreddit", "post_id", "parent_id", "post_score",
          "num_comments", "vader_compound", "vader_label",
          "textblob_polarity", "textblob_subjectivity", "textblob_label"]
Y_COLS = ["index", "Label", "text", "likeCount", "created_time", "video_date",
          "video id", "author", "vader_compound", "vader_label",
          "textblob_polarity", "textblob_subjectivity", "textblob_label"]

STANCES = ["P", "I", "N"]


def main():
    print("Loading Reddit sentiment data ...")
    r = pd.read_csv(SENT / "reddit_with_sentiment.csv", usecols=R_COLS)
    print(f"  loaded {len(r):,}")
    r["created_time"] = pd.to_datetime(r["created_time"], errors="coerce")
    r = r[r["Label"].isin(STANCES)]

    n_before = len(r)
    r_win = r[(r["created_time"] >= WINDOW_START) &
              (r["created_time"] <= WINDOW_END + pd.Timedelta(days=1))].copy()
    dropped = n_before - len(r_win)
    print(f"  window filter: {n_before:,} -> {len(r_win):,} "
          f"(dropped {dropped:,} = {dropped / n_before * 100:.2f}% pre-window)")

    print("Loading YouTube sentiment data ...")
    y = pd.read_csv(SENT / "youtube_with_sentiment.csv", usecols=Y_COLS)
    print(f"  loaded {len(y):,}")
    y["video_date"] = pd.to_datetime(y["video_date"], errors="coerce")
    y = y[y["Label"].isin(STANCES)]

    n_before_y = len(y)
    y_win = y[(y["video_date"] >= WINDOW_START) &
              (y["video_date"] <= WINDOW_END + pd.Timedelta(days=1))].copy()
    dropped_y = n_before_y - len(y_win)
    print(f"  window filter: {n_before_y:,} -> {len(y_win):,} "
          f"(dropped {dropped_y:,} = {dropped_y / max(n_before_y,1) * 100:.2f}%)")

    # Derived features used downstream
    r_win["word_count"] = r_win["self_text"].fillna("").str.split().str.len()
    y_win["word_count"] = y_win["text"].fillna("").str.split().str.len()
    # Unified temporal anchor for each platform
    r_win["time_anchor"] = r_win["created_time"]
    y_win["time_anchor"] = y_win["video_date"]
    # Unified text column
    r_win["text_std"] = r_win["self_text"].fillna("")
    y_win["text_std"] = y_win["text"].fillna("")

    print("Writing working files ...")
    r_win.to_parquet(OUT / "reddit_window.parquet", index=False)
    y_win.to_parquet(OUT / "youtube_window.parquet", index=False)

    summary = {
        "window_start": str(WINDOW_START.date()),
        "window_end": str(WINDOW_END.date()),
        "reddit_before": n_before, "reddit_after": len(r_win), "reddit_dropped": dropped,
        "youtube_before": n_before_y, "youtube_after": len(y_win), "youtube_dropped": dropped_y,
    }
    pd.Series(summary).to_csv(OUT / "window_summary.csv")
    print("\n--- WINDOW SUMMARY ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nStance composition (windowed):")
    for name, df in [("Reddit", r_win), ("YouTube", y_win)]:
        vc = df["Label"].value_counts().reindex(STANCES)
        pct = (vc / vc.sum() * 100).round(2)
        print(f"  {name}: " + ", ".join(f"{s}={vc[s]:,} ({pct[s]}%)" for s in STANCES))


if __name__ == "__main__":
    main()
