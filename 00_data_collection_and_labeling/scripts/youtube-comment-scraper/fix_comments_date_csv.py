"""
fix_comments_csv.py

Cleans up youtube_war_comments.csv:
  1. Removes the `comment_updated_at` column.
  2. Converts `comment_published_at` from relative strings ("X years ago",
     "X months ago", etc.) to approximate calendar dates.
     Column is renamed to `comment_published_at_approx`.
  3. Adds a `video_date` column containing the publish date of the
     corresponding video (from youtube_video_metadata.csv).

Output: youtube_war_comments_fixed_date.csv
"""

import csv
import re
from datetime import date, timedelta
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
IN_COMMENTS  = BASE / "youtube_war_comments.csv"
IN_METADATA  = BASE / "youtube_video_metadata.csv"
OUT_COMMENTS = BASE / "youtube_war_comments_fixed_date.csv"

# ── reference date for relative-time conversion ──────────────────────────────
# Using today's date as the reference point. Dates are approximate because
# the original scraper only recorded relative strings ("X years ago"), not
# exact timestamps.
REFERENCE_DATE = date.today()


def parse_relative_date(rel: str) -> str:
    """
    Convert a YouTube-style relative time string to an approximate date.

    Handles patterns like:
      "just now", "1 hour ago", "3 hours ago",
      "1 day ago",  "5 days ago",
      "2 weeks ago",
      "1 month ago", "8 months ago",
      "1 year ago",  "3 years ago"

    Returns an ISO date string "YYYY-MM-DD" or the original string if it
    cannot be parsed.
    """
    if not rel or not isinstance(rel, str):
        return ""

    s = rel.strip().lower()

    if s in ("just now", "moments ago"):
        return REFERENCE_DATE.isoformat()

    match = re.match(
        r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", s
    )
    if not match:
        return rel  # return as-is if unrecognised

    value = int(match.group(1))
    unit  = match.group(2)

    if unit in ("second", "minute", "hour"):
        # sub-day precision – treat as today
        delta = timedelta(days=0)
    elif unit == "day":
        delta = timedelta(days=value)
    elif unit == "week":
        delta = timedelta(weeks=value)
    elif unit == "month":
        # approximate: 1 month ≈ 30 days
        delta = timedelta(days=value * 30)
    elif unit == "year":
        # approximate: 1 year ≈ 365 days
        delta = timedelta(days=value * 365)
    else:
        return rel

    approx = REFERENCE_DATE - delta
    return approx.isoformat()


# ── load video dates ──────────────────────────────────────────────────────────
video_date: dict[str, str] = {}
with open(IN_METADATA, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        vid = row["video_id"].strip()
        raw = row.get("published_at", "").strip()
        # published_at is ISO-8601: "2023-10-19T11:15:07Z" → keep date part only
        video_date[vid] = raw[:10] if raw else ""

# ── process comments ──────────────────────────────────────────────────────────
with (
    open(IN_COMMENTS,  newline="", encoding="utf-8") as fin,
    open(OUT_COMMENTS, "w", newline="", encoding="utf-8") as fout,
):
    reader = csv.DictReader(fin)

    # Build output fieldnames:
    #   • drop  `comment_updated_at`
    #   • rename `comment_published_at` → `comment_published_at_approx`
    #   • append `video_date`
    in_fields = reader.fieldnames or []
    out_fields = []
    for col in in_fields:
        if col == "comment_updated_at":
            continue
        elif col == "comment_published_at":
            out_fields.append("comment_published_at_approx")
        else:
            out_fields.append(col)
    out_fields.append("video_date")

    writer = csv.DictWriter(fout, fieldnames=out_fields)
    writer.writeheader()

    for row in reader:
        out_row: dict[str, str] = {}
        for col in in_fields:
            if col == "comment_updated_at":
                continue
            elif col == "comment_published_at":
                out_row["comment_published_at_approx"] = parse_relative_date(
                    row[col]
                )
            else:
                out_row[col] = row[col]

        vid = row.get("video_id", "").strip()
        out_row["video_date"] = video_date.get(vid, "")

        writer.writerow(out_row)

print(f"Done. Written to: {OUT_COMMENTS}")
print(f"Reference date used for relative-time conversion: {REFERENCE_DATE}")
print(f"Video dates loaded for {len(video_date)} video IDs.")
