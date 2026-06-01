"""
Backfill YouTube war dataset for August and September 2023,
then merge into the existing datasets that currently start from October 2023.

This script reuses the same scraping logic from yt_scrape_optimized.py.
Backfill rows are written to temporary CSVs first, then merged and deduplicated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd
from googleapiclient.errors import HttpError

import yt_scrape_optimized as scraper


BASE = Path(__file__).resolve().parent

BACKFILL_EXTRA_QUERIES = [
    "israel palestine conflict",
    "israel palestine latest",
    "israel palestine news",
    "gaza israel conflict",
    "gaza strip news",
    "hamas israel tensions",
    "west bank clashes",
    "west bank violence",
    "jenin raid",
    "middle east tensions",
    "palestine israel tension",
    "palestine israel violence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Aug-Sep 2023 data and merge into existing datasets."
    )
    parser.add_argument(
        "--published-after",
        default="2023-08-01T00:00:00Z",
        help="Inclusive lower bound in ISO-8601 UTC format.",
    )
    parser.add_argument(
        "--published-before",
        default="2023-09-30T23:59:59Z",
        help="Inclusive upper bound in ISO-8601 UTC format.",
    )
    parser.add_argument(
        "--original-comments-file",
        default=str(BASE / "youtube_war_comments.csv"),
        help="Main comments CSV to extend.",
    )
    parser.add_argument(
        "--original-metadata-file",
        default=str(BASE / "youtube_video_metadata.csv"),
        help="Main metadata CSV to extend.",
    )
    parser.add_argument(
        "--backfill-comments-file",
        default=str(BASE / "youtube_war_comments_aug_sep_2023.csv"),
        help="Temporary backfill comments output CSV.",
    )
    parser.add_argument(
        "--backfill-metadata-file",
        default=str(BASE / "youtube_video_metadata_aug_sep_2023.csv"),
        help="Temporary backfill metadata output CSV.",
    )
    parser.add_argument(
        "--backfill-processed-file",
        default=str(BASE / "processed_videos_aug_sep_2023.txt"),
        help="Temporary processed videos tracker for this backfill window.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=2,
        help="YouTube search pages per query for backfill discovery.",
    )
    parser.add_argument(
        "--search-order",
        default="date",
        choices=["date", "viewCount", "relevance", "rating", "title"],
        help="YouTube API search order for backfill discovery.",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping and only run merge/dedup using existing backfill files.",
    )
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _build_backfill_queries() -> list[str]:
    return _dedupe_preserve_order(list(scraper.QUERIES) + BACKFILL_EXTRA_QUERIES)


def _make_backfill_search_function(args: argparse.Namespace):
    def _search_videos_backfill(query: str, max_pages: int = 6) -> list[str]:
        video_ids: list[str] = []
        token = None

        for _ in range(args.max_pages):
            try:
                res = scraper.youtube.search().list(
                    part="id",
                    q=query,
                    type="video",
                    order=args.search_order,
                    publishedAfter=scraper.PUBLISHED_AFTER,
                    publishedBefore=scraper.PUBLISHED_BEFORE,
                    maxResults=50,
                    pageToken=token,
                ).execute()

                for item in res.get("items", []):
                    vid = item.get("id", {}).get("videoId")
                    if vid:
                        video_ids.append(vid)

                token = res.get("nextPageToken")
                if not token:
                    break

            except HttpError as e:
                if e.resp.status == 403 and "quota" in str(e).lower():
                    print(f"\n⚠️  Quota exceeded during search for '{query}'")
                    raise
                print(f"Error searching '{query}': {e}")
                break

            time.sleep(0.2)

        return video_ids

    return _search_videos_backfill


def _dedupe_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "video_id" in out.columns:
        out = out.drop_duplicates(subset=["video_id"], keep="first")
    if "published_at" in out.columns:
        out["_published_at_dt"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
        out = out.sort_values(by="_published_at_dt", kind="stable").drop(columns=["_published_at_dt"])
    return out.reset_index(drop=True)


def _dedupe_comments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if "comment_id" not in out.columns:
        return out.drop_duplicates(keep="first").reset_index(drop=True)

    comment_id = out["comment_id"].astype(str).str.strip()

    fallback_parts = [
        out.get("video_id", "").astype(str),
        out.get("author", "").astype(str),
        out.get("comment_text", "").astype(str),
        out.get("comment_published_at", "").astype(str),
    ]
    fallback_key = fallback_parts[0]
    for part in fallback_parts[1:]:
        fallback_key = fallback_key + "|" + part

    dedupe_key = comment_id.where(comment_id != "", fallback_key)
    out = out.loc[~dedupe_key.duplicated(keep="first")]
    return out.reset_index(drop=True)


def _merge_and_write(
    original_path: Path,
    backfill_path: Path,
    dedupe_fn,
    label: str,
) -> None:
    original_df = _read_csv_if_exists(original_path)
    backfill_df = _read_csv_if_exists(backfill_path)

    if backfill_df.empty:
        print(f"No backfill {label} rows found at: {backfill_path}")
        return

    original_dedup = dedupe_fn(original_df)
    combined_df = pd.concat([original_dedup, backfill_df], ignore_index=True, sort=False)
    merged_df = dedupe_fn(combined_df)

    if not original_df.empty:
        existing_columns = list(original_df.columns)
        merged_columns = existing_columns + [c for c in merged_df.columns if c not in existing_columns]
        merged_df = merged_df[merged_columns]

    original_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(original_path, index=False, encoding="utf-8")

    added_rows = len(merged_df) - len(original_dedup)
    removed_existing_duplicates = len(original_df) - len(original_dedup)
    print(
        f"Merged {label}: +{added_rows} net new rows, "
        f"removed {removed_existing_duplicates} existing duplicates."
    )
    print(f"Final {label} rows: {len(merged_df)}")


def run_backfill_scrape(args: argparse.Namespace) -> None:
    scraper.PUBLISHED_AFTER = args.published_after
    scraper.PUBLISHED_BEFORE = args.published_before
    scraper.COMMENTS_FILE = args.backfill_comments_file
    scraper.METADATA_FILE = args.backfill_metadata_file
    scraper.PROCESSED_FILE = args.backfill_processed_file
    scraper.QUERIES = _build_backfill_queries()
    scraper.search_videos = _make_backfill_search_function(args)

    print("=" * 70)
    print("Running backfill scrape")
    print(f"Date window: {args.published_after} to {args.published_before}")
    print(f"Search order: {args.search_order}")
    print(f"Pages/query: {args.max_pages}")
    print(f"Queries used: {len(scraper.QUERIES)}")
    print(f"Backfill metadata: {args.backfill_metadata_file}")
    print(f"Backfill comments: {args.backfill_comments_file}")
    print(f"Backfill processed tracker: {args.backfill_processed_file}")
    print("=" * 70)
    scraper.main()


def main() -> None:
    args = parse_args()

    original_comments = Path(args.original_comments_file)
    original_metadata = Path(args.original_metadata_file)
    backfill_comments = Path(args.backfill_comments_file)
    backfill_metadata = Path(args.backfill_metadata_file)

    if not args.skip_scrape:
        run_backfill_scrape(args)

    print("\nMerging backfill data into original datasets...")
    _merge_and_write(original_metadata, backfill_metadata, _dedupe_metadata, "metadata")
    _merge_and_write(original_comments, backfill_comments, _dedupe_comments, "comments")

    print("\nDone. Original datasets have been extended with Aug-Sep 2023 data.")


if __name__ == "__main__":
    main()
