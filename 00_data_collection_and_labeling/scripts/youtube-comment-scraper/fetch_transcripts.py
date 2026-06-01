"""
fetch_transcripts.py
--------------------
Fetches YouTube transcripts for every video_id in youtube_video_metadata.csv.

Outputs
-------
  transcripts.csv          – video_id, language_code, transcript (full text)
  no_transcript_ids.csv    – video_id for videos with no available captions

Anti-blocking strategy
----------------------
  youtube-transcript-api v1.2+ blocks direct requests from many IPs.
  The RECOMMENDED workaround (per the library's README) is to pass your
  authenticated YouTube session cookies — this makes requests look like
  they come from a real, logged-in browser.

  How to export your cookies:
    1. Install the browser extension "Get cookies.txt LOCALLY"
       (Chrome: https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
       (Firefox: similar extension available)
    2. Go to https://www.youtube.com and make sure you're logged in.
    3. Click the extension → "Export" → choose "Current Site" → save as
       youtube_cookies.txt in this directory.
    4. Pass --cookies youtube_cookies.txt when running this script.

  Additional measures:
    • Random jittered sleep between requests [MIN_SLEEP, MAX_SLEEP] seconds.
    • Longer batch pause every BATCH_SIZE requests.
    • Exponential backoff retries on transient errors.
    • Fully resumable — already-processed IDs are skipped.

Usage
-----
  # With cookies (strongly recommended for 2.6k videos):
  python fetch_transcripts.py --cookies youtube_cookies.txt

  # Without cookies (will likely be blocked quickly):
  python fetch_transcripts.py
"""

import argparse
import csv
import http.cookiejar
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV       = "youtube_video_metadata.csv"
OUTPUT_CSV      = "transcripts.csv"
MISSING_CSV     = "no_transcript_ids.csv"

MIN_SLEEP       = 2.0   # seconds between requests (lower bound)
MAX_SLEEP       = 5.0   # seconds between requests (upper bound)
BATCH_SIZE      = 40    # after every N requests, take a longer break
BATCH_PAUSE_MIN = 20    # longer pause lower bound (seconds)
BATCH_PAUSE_MAX = 45    # longer pause upper bound (seconds)

MAX_RETRIES     = 3     # retries on transient / rate-limit errors
RETRY_BACKOFF   = 15    # base seconds to wait on first retry (doubles each time)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Cookie loading ────────────────────────────────────────────────────────────
def build_session_with_cookies(cookie_file: str) -> requests.Session:
    """
    Load a Netscape/Mozilla cookies.txt file into a requests.Session.
    This lets the API authenticate as your browser, bypassing IP blocks.
    """
    session = requests.Session()
    cj = http.cookiejar.MozillaCookieJar()

    cookie_path = Path(cookie_file)
    if not cookie_path.exists():
        raise FileNotFoundError(
            f"Cookie file not found: {cookie_file}\n"
            "See the script docstring for how to export cookies."
        )

    cj.load(str(cookie_path), ignore_discard=True, ignore_expires=True)
    session.cookies = cj  # type: ignore[assignment]

    # Mimic a real browser
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    log.info("Loaded cookies from %s (%d cookies)", cookie_file, len(list(cj)))
    return session


# ── Already-processed tracking ────────────────────────────────────────────────
def load_already_processed(output_csv: str, missing_csv: str) -> set:
    """Return set of video_ids already written to either output file."""
    seen = set()
    for path in (output_csv, missing_csv):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "video_id" in df.columns:
                    seen.update(df["video_id"].dropna().astype(str).tolist())
            except Exception:
                pass
    return seen


# ── Transcript fetching ───────────────────────────────────────────────────────
def segments_to_text(segments) -> str:
    """Join FetchedTranscript segments into a single clean string."""
    parts = []
    for seg in segments:
        text = getattr(seg, "text", None) or (seg.get("text") if isinstance(seg, dict) else "")
        if text:
            parts.append(text.strip())
    return " ".join(parts)


def fetch_best_transcript(api: YouTubeTranscriptApi, video_id: str):
    """
    Try to fetch the best available transcript for *video_id*.

    Priority: manually-created > auto-generated, any language.

    Returns (language_code, transcript_text) on success.
    Returns (None, None) if no transcript exists at all.
    Raises for transient/unexpected errors so caller can retry.
    """
    transcript_list = api.list(video_id)

    chosen = None

    # 1. Prefer manually created transcripts (any language)
    for t in transcript_list:
        if not t.is_generated:
            chosen = t
            break

    # 2. Fall back to auto-generated (any language)
    if chosen is None:
        for t in transcript_list:
            chosen = t
            break

    if chosen is None:
        return None, None

    segments = chosen.fetch()
    return chosen.language_code, segments_to_text(segments)


# ── CSV writing ───────────────────────────────────────────────────────────────
def write_row(path: str, row: dict, fieldnames: list):
    """Append a single row; writes header if file is new."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube transcripts.")
    parser.add_argument(
        "--cookies",
        metavar="FILE",
        default=None,
        help="Path to a Netscape cookies.txt file exported from your browser. "
             "Strongly recommended to avoid IP blocks on large batches.",
    )
    parser.add_argument(
        "--input",  default=INPUT_CSV,  help=f"Input CSV (default: {INPUT_CSV})"
    )
    parser.add_argument(
        "--output", default=OUTPUT_CSV, help=f"Output CSV (default: {OUTPUT_CSV})"
    )
    parser.add_argument(
        "--missing", default=MISSING_CSV, help=f"Missing IDs CSV (default: {MISSING_CSV})"
    )
    args = parser.parse_args()

    # Build API instance
    if args.cookies:
        session = build_session_with_cookies(args.cookies)
        api = YouTubeTranscriptApi(http_client=session)
        log.info("Using cookie-authenticated session.")
    else:
        api = YouTubeTranscriptApi()
        log.warning(
            "No cookies provided. For 2.6k videos this will almost certainly "
            "trigger an IP block. Run with --cookies youtube_cookies.txt."
        )

    # Load IDs
    log.info("Loading video IDs from %s …", args.input)
    df = pd.read_csv(args.input)
    video_ids = df["video_id"].dropna().astype(str).unique().tolist()
    log.info("Total unique video IDs: %d", len(video_ids))

    already_done = load_already_processed(args.output, args.missing)
    remaining = [v for v in video_ids if v not in already_done]
    log.info(
        "Already processed: %d  |  Remaining: %d",
        len(already_done), len(remaining),
    )

    if not remaining:
        log.info("Nothing left to process. Exiting.")
        return

    success_count = 0
    missing_count = 0
    error_count   = 0

    for i, video_id in enumerate(tqdm(remaining, desc="Fetching transcripts", unit="video")):

        lang_code = None
        transcript_text = None
        is_missing = False

        # ── Fetch with retries ────────────────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                lang_code, transcript_text = fetch_best_transcript(api, video_id)
                if transcript_text is None:
                    is_missing = True
                break   # success or definitively missing

            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
                is_missing = True
                break   # no point retrying

            except (IpBlocked, RequestBlocked) as e:
                # Serious block — back off hard
                wait = RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 10)
                log.warning(
                    "[%s] IP/Request blocked (attempt %d/%d) — sleeping %.0fs before retry.",
                    video_id, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)

            except CouldNotRetrieveTranscript as e:
                wait = RETRY_BACKOFF * attempt + random.uniform(0, 5)
                log.warning(
                    "[%s] CouldNotRetrieve (attempt %d/%d): %s — retrying in %.0fs",
                    video_id, attempt, MAX_RETRIES, str(e)[:80], wait,
                )
                time.sleep(wait)

            except Exception as e:
                wait = RETRY_BACKOFF * attempt + random.uniform(0, 5)
                log.warning(
                    "[%s] Unexpected error (attempt %d/%d): %s — retrying in %.0fs",
                    video_id, attempt, MAX_RETRIES, str(e)[:120], wait,
                )
                time.sleep(wait)
        else:
            # All retries exhausted — don't write so this ID is retried next run
            log.error("[%s] All %d retries failed — will retry on next run.", video_id, MAX_RETRIES)
            error_count += 1
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            continue

        # ── Write result ──────────────────────────────────────────────────────
        if is_missing:
            write_row(args.missing, {"video_id": video_id}, ["video_id"])
            missing_count += 1
        else:
            write_row(
                args.output,
                {"video_id": video_id, "language_code": lang_code, "transcript": transcript_text},
                ["video_id", "language_code", "transcript"],
            )
            success_count += 1

        # ── Anti-blocking sleep ───────────────────────────────────────────────
        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        # Longer batch pause every BATCH_SIZE videos
        if (i + 1) % BATCH_SIZE == 0:
            pause = random.uniform(BATCH_PAUSE_MIN, BATCH_PAUSE_MAX)
            log.info(
                "Batch pause after %d videos — sleeping %.0fs …", i + 1, pause
            )
            time.sleep(pause)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Transcripts saved   : %d  → %s", success_count, args.output)
    log.info("  No transcript found : %d  → %s", missing_count, args.missing)
    log.info("  Errors (will retry) : %d  (re-run script to retry)", error_count)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
