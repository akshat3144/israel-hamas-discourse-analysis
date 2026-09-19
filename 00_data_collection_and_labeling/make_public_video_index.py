# -*- coding: utf-8 -*-
"""Produce the releasable video index from the full metadata file.

The full file carries video titles and descriptions retrieved through the
YouTube Data API. The API Services Developer Policies (III.E.4.c) cap retention
of that data at 30 days and III.E.3.b restricts sharing it, with no research
exception, so titles and descriptions are not redistributed here.

What is released is the identifier set: anyone can re-fetch the titles and
descriptions for these 2,637 videos with a single videos.list call each and
reproduce the video-context experiment in Section 3.4.3.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "outputs" / "youtube_video_metadata.csv"
OUT = HERE / "outputs" / "youtube_video_index.csv"

KEEP = ["video_id", "channel_id", "channel_title", "published_at"]
WITHHELD = ["title", "description"]


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    d = pd.read_csv(SRC, low_memory=False)

    missing = [c for c in KEEP if c not in d.columns]
    if missing:
        sys.exit(f"expected columns absent: {missing}")

    out = d[KEEP].drop_duplicates(subset="video_id")
    out.to_csv(OUT, index=False)

    print(f"  source   : {SRC.name}  ({len(d):,} rows, {SRC.stat().st_size/1e6:.2f} MB)")
    print(f"  released : {OUT.name}  ({len(out):,} videos, "
          f"{OUT.stat().st_size/1e6:.2f} MB)")
    print(f"  withheld : {', '.join(WITHHELD)}  (YouTube Developer Policies)")


if __name__ == "__main__":
    main()
