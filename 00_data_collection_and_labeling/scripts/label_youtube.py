"""
youtube Data Labeling Script using dcompute (OpenAI-compatible API)
Model: meta-llama/Llama-3.3-70B-Instruct
"""

import json
import os
import asyncio
import tempfile
import re
import csv
import shutil
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# API configuration
DCOMPUTE_BASE_URL = os.getenv(
    "DCOMPUTE_BASE_URL"
)
DCOMPUTE_API_KEY = os.getenv("DCOMPUTE_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("DCOMPUTE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# Runtime configuration
BATCH_SIZE = int(os.getenv("LABEL_BATCH_SIZE", "10"))
MAX_RETRIES = int(os.getenv("LABEL_MAX_RETRIES", "10"))
TEMPERATURE = float(os.getenv("LABEL_TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS = int(os.getenv("LABEL_MAX_OUTPUT_TOKENS", "2048"))

# File paths
INPUT_FILE = os.getenv("YOUTUBE_INPUT_FILE", "data/youtube.csv")
OUTPUT_FILE = os.getenv("YOUTUBE_OUTPUT_FILE", "data/youtube_labeled_full.jsonl")
PROGRESS_FILE = os.getenv("YOUTUBE_PROGRESS_FILE", "youtube_progress_full.json")
BATCHES_DIR = os.getenv("YOUTUBE_BATCHES_DIR", "").strip()
WAVES_DIR = os.path.join(os.path.dirname(OUTPUT_FILE) or ".", ".waves_youtube")


def load_annotation_guidelines():
    """Load annotation guidelines from PDF or text file"""
    guidelines = """
    ANNOTATION GUIDELINES FOR ISRAEL-GAZA WAR DISCOURSE
    
    Objective: Label each datapoint with its corresponding stance towards the Israel-Gaza conflict.
    
    STANCE LABELS:
    
    P = Supports Palestine
    - Advocates for Palestinian rights, interests, or perspectives
    - Supports Palestinian statehood, sovereignty, self-determination, independence, and equality
    - Criticizes Israel's actions or policies towards Palestinians
    - Examples:
      * "Hamas wanted to negotiate the week of the 7th. USA/Israel said no."
      * "Not all Palestinians support Hamas. However, being able to express it leads to death in Gaza."
      * "It is more than just Israel. It is the imperialist countries, too, including the United States."
    
    I = Supports Israel
    - Supportive of Israel's interests, security, and rights
    - Backs Israel's sovereignty, territorial integrity, and protection of citizens
    - Supports Israel's right to defend itself and ensure survival
    - Examples:
      * "It's the ultimate gaslighting to blame Israel for people deciding to be terrorists."
      * "Israel doesn't occupy Gaza, for the past 18 years. They pulled settlers and military out in 2005."
      * "Thank you Israel. Taking the garbage out now so we don't have to in 5 years."
    
    N = Neutral/Unclear Stance
    - Impartial or ambiguous viewpoint
    - No definitive position favoring either Palestinian or Israeli side
    - Lacks sufficient information or intentionally avoids partisanship
    - Presents balanced views or asks questions without taking sides
    - Examples:
      * "There's a pretty large difference between engineers making nuclear weapons vs. random Israeli civilians."
      * "All I ask is for my legitimate questions to be legitimately answered."

    R = Irrelevant / Not Related
    - Not related to the Israel-Gaza conflict
    - Off-topic, spam, jokes, memes, or unrelated discussions
    - Generic political or social commentary without reference to the conflict
    - Too vague to determine relevance
    - Examples:
        * "This subreddit has gone downhill lately."
        * "I just got a new phone today!"
        * "Politics is always messy everywhere."
    
    IMPORTANT INSTRUCTIONS:
    - Evaluate based on context, tone, and content
    - Consider nuances and subtleties in language
    - Avoid assumptions or bias
    - Some datapoints may be incomplete (part of ongoing conversation)
    
    Provide labels in JSON format with fields: "Label" (must be P, I, N, or R), "Confidence", "Reasoning"
    """
    return guidelines


def safe_field(row, candidates, default="N/A", max_chars=None):
    """Return first non-empty field from candidate column names."""
    for key in candidates:
        if key in row and pd.notna(row[key]):
            value = str(row[key])
            if value.strip() == "":
                continue
            if max_chars is not None:
                return value[:max_chars]
            return value
    return default


def create_prompt_for_batch(batch_df, guidelines):

    records = batch_df if isinstance(batch_df, list) else batch_df.to_dict(orient="records")

    lines = [
f"""You are an expert annotator analyzing discourse about the Israel-Hamas war on YouTube.

{guidelines}

Analyze the following {len(records)} YouTube comments.

For each item consider:
- Comment text
- Video context
- Author
- Metadata if available

Data:
"""
    ]

    for i, row in enumerate(records):

        idx = row.get("__row_index", row.get("index", i))

        comment_id = safe_field(
            row,
            ["id","comment_id"]
        )

        video_id = safe_field(
            row,
            ["video id","video_id"]
        )

        author = safe_field(
            row,
            ["author","author_name","channel"]
        )

        comment = safe_field(
            row,
            ["text","comment","body"],
            max_chars=1200
        )

        likes = safe_field(
            row,
            ["likes","like_count"],
            default="N/A"
        )

        lines.append(f"""
--- Item {idx} ---

Comment ID: {comment_id}
Video ID: {video_id}
Author: {author}
Likes: {likes}

Comment:
{comment}
""")

    lines.append(
f"""

Return EXACTLY {len(records)} JSON objects.

[
{{
"index":0,
"Label":"P",
"Confidence":"High",
"Reasoning":"..."
}}
]

Allowed labels only:
P
I
N
R

Respond ONLY JSON.
"""
    )

    return "\n".join(lines)
def parse_model_response(response_text):
    """Parse model response and extract label JSON."""
    text = response_text.strip()
    candidates = []

    if "```json" in text:
        try:
            candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
        except Exception:
            pass
    if "```" in text:
        try:
            candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())
        except Exception:
            pass

    candidates.append(text)
    first_array = text.find("[")
    last_array = text.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        candidates.append(text[first_array:last_array + 1].strip())

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c and c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    for candidate in unique_candidates:
        try:
            parsed = json.loads(candidate, strict=False)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                if isinstance(parsed.get("labels"), list):
                    return parsed["labels"]
                if isinstance(parsed.get("data"), list):
                    return parsed["data"]
        except Exception:
            pass

        # Cleanup pass for common malformed JSON patterns.
        try:
            json_str_cleaned = candidate
            json_str_cleaned = re.sub(r',\s*([}\]])', r'\1', json_str_cleaned)
            json_str_cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str_cleaned)
            parsed = json.loads(json_str_cleaned, strict=False)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                if isinstance(parsed.get("labels"), list):
                    return parsed["labels"]
                if isinstance(parsed.get("data"), list):
                    return parsed["data"]
        except Exception:
            continue

    print("Error parsing response after all recovery attempts")
    print(f"Response text: {response_text[:500]}")
    return None


async def label_batch_with_llama(client, batch_df, guidelines):
    """Label a batch using Llama 3.3 70B with bounded retries for transient errors."""
    prompt = create_prompt_for_batch(batch_df, guidelines)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            )

            text = response.choices[0].message.content if response and response.choices else ""
            print("received response")
            if not text:
                raise ValueError("Empty response from model")

            labels = parse_model_response(text)
            if labels is None:
                raise ValueError("Failed to parse model output as JSON")

            return labels
        except Exception as e:
            backoff = min(2 * attempt, 30)
            print(f"Error calling dcompute API (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise
            await asyncio.sleep(backoff)


async def process_batch(client, batch_df, guidelines):
    """Process one batch with shared async API client."""
    return await label_batch_with_llama(client, batch_df, guidelines)


async def run_batch(semaphore, client, batch_payload, guidelines, batch_start, batch_end, use_folder_mode=False):
    """Run a single batch under semaphore-based concurrency control (or unrestricted if None).
    Returns: batch_start, batch_end, batch_df, labels, error_msg"""
    async def do_batch():
        try:
            batch_df = load_batch_dataframe(batch_payload) if use_folder_mode else batch_payload
            labels = await process_batch(client, batch_df, guidelines)
            return batch_start, batch_end, batch_df, labels, None
        except Exception as e:
            return batch_start, batch_end, None, None, str(e)
    
    if semaphore is None:
        return await do_batch()
    else:
        async with semaphore:
            return await do_batch()


def save_progress(completed_waves, total_waves):
    """Save progress to file atomically."""
    progress = {
        'completed_waves': sorted(list(completed_waves)),
        'total_waves': total_waves,
        'last_updated': datetime.now().isoformat()
    }
    progress_dir = os.path.dirname(PROGRESS_FILE) or "."
    os.makedirs(progress_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="progress_", suffix=".tmp", dir=progress_dir)
    os.close(fd)
    with open(temp_path, 'w') as f:
        json.dump(progress, f)
    os.replace(temp_path, PROGRESS_FILE)


def load_progress():
    """Load progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            return {'completed_waves': set(data.get('completed_waves', []))}
    return {'completed_waves': set()}


def load_dataframe(path):
    """Load CSV or Excel input files."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path}")


def load_batches_from_manifest(batches_dir):
    """Load batch descriptors from manifest.json without listing files."""
    manifest_path = os.path.join(batches_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise ValueError(f"manifest.json not found in {batches_dir}. Re-run prepare_reddit_batches.py")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    batch_size = manifest["batch_size"]
    total_rows = manifest["total_rows"]
    
    # Compute batch ranges algorithmically without file listing
    descriptors = []
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_path = os.path.join(batches_dir, f"batch_{batch_start:010d}_{batch_end - 1:010d}.csv")
        descriptors.append((batch_start, batch_end, batch_path))
    
    return descriptors


def load_batch_dataframe(path):
    """Load one pre-split batch CSV as list-of-dicts for faster tiny-file parsing."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "__row_index" in row:
                try:
                    row["__row_index"] = int(row["__row_index"])
                except Exception:
                    pass
            rows.append(row)
    return rows


def prepare_output_file(start_idx):
    """Prepare append-only output file based on resume state."""
    output_dir = os.path.dirname(OUTPUT_FILE) or "."
    os.makedirs(output_dir, exist_ok=True)
    if start_idx == 0 and os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    open(OUTPUT_FILE, "a", encoding="utf-8").close()


def append_batch_results(batch_records, wave_idx=None):
    """Append one JSON object per labeled row as newline-delimited JSON.
    If wave_idx is provided, write to wave-specific file; otherwise write to final output."""
    if wave_idx is not None:
        os.makedirs(WAVES_DIR, exist_ok=True)
        output_path = os.path.join(WAVES_DIR, f"wave_{wave_idx:02d}.jsonl")
    else:
        output_path = OUTPUT_FILE
    
    with open(output_path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in batch_records)


def merge_wave_files(num_waves):
    """Merge all wave_XX.jsonl files in order into final OUTPUT_FILE atomically."""
    print(f"\nMerging {num_waves} wave files...")
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    # Write to a temp file first, then atomically swap — crash-safe
    tmp_output = OUTPUT_FILE + ".merging.tmp"
    with open(tmp_output, "w", encoding="utf-8") as out_f:
        for wave_idx in range(num_waves):
            wave_file = os.path.join(WAVES_DIR, f"wave_{wave_idx:02d}.jsonl")
            if os.path.exists(wave_file):
                with open(wave_file, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
            else:
                print(f"  WARNING: wave file missing for wave {wave_idx}, skipping")

    # Atomic replace — only succeeds if the write above fully completed
    os.replace(tmp_output, OUTPUT_FILE)

    # Only clean up wave files AFTER the atomic swap confirms success
    if os.path.exists(WAVES_DIR):
        shutil.rmtree(WAVES_DIR)

    print(f"✓ Merged all waves → {OUTPUT_FILE}")


semaphore = asyncio.Semaphore(5)


async def process_wave(wave_idx, wave_batches, client, guidelines):
    """Process one independent wave of batches. Returns wave_idx and stats."""
    import time
    wave_batch_records = []
    wave_success = 0
    wave_failed = 0
    failed_batches = []
    wave_start_time = time.time()
    
    print(f"  [Wave {wave_idx}] Starting: {len(wave_batches)} batches", flush=True)
    
    # Process all batches in this wave concurrently
    tasks = [
        asyncio.create_task(
            run_batch(
                semaphore,
                client,
                batch_payload,
                guidelines,
                batch_start,
                batch_end,
                use_folder_mode=bool(BATCHES_DIR),
            )
        )
        for batch_start, batch_end, batch_payload in wave_batches
    ]
    
    # Collect results as they complete
    completed_count = 0
    for task in asyncio.as_completed(tasks):
        batch_start, batch_end, batch_df, labels, error_msg = await task
        completed_count += 1
        
        if error_msg is None and batch_df is not None:
            batch_records = []
            csv_rows = batch_df if isinstance(batch_df, list) else batch_df.to_dict('records')
            csv_by_idx = {i: row for i, row in enumerate(csv_rows)}
            
            for label_data in labels:
                try:
                    idx = label_data['index']
                    label = label_data.get('Label', 'N')
                    if label not in ['P', 'I', 'N', 'R']:
                        label = 'N'
                    if idx < batch_start or idx >= batch_end:
                        continue
                    confidence = label_data.get('Confidence', '')
                    reasoning = label_data.get('Reasoning', '')
                    
                    row_offset = next(
                        (
                        j
                        for j,r in enumerate(csv_rows)
                        if int(r["__row_index"]) == idx
                        ),
                        None
                        )
                    csv_row = csv_by_idx.get(row_offset, {})
                    
                    record = {
                        'index': idx,
                        'Label': label,
                        'Confidence': confidence,
                        'Reasoning': reasoning
                    }
                    if isinstance(csv_row, dict):
                        record.update(csv_row)
                    batch_records.append(record)
                except Exception:
                    continue
            
            batch_records.sort(key=lambda x: x['index'])
            expected_count = batch_end - batch_start
            if len(batch_records) != expected_count:
                failed_batches.append((batch_start, batch_end, "Label count mismatch"))
                wave_failed += 1
                continue
            
            wave_success += 1
            wave_batch_records.extend(batch_records)
            print(f"  [Wave {wave_idx}] Batch {batch_start}-{batch_end}: ✓ ({completed_count}/{len(tasks)})", flush=True)
        else:
            failed_batches.append((batch_start, batch_end, error_msg))
            wave_failed += 1
            print(f"  [Wave {wave_idx}] Batch {batch_start}-{batch_end}: ✗ {error_msg} ({completed_count}/{len(tasks)})", flush=True)
    
    # Sort and write wave results to wave-specific file
    wave_batch_records.sort(key=lambda x: x['index'])
    if wave_batch_records:
        append_batch_results(wave_batch_records, wave_idx=wave_idx)
    
    elapsed = time.time() - wave_start_time
    print(f"  [Wave {wave_idx}] Complete: {wave_success} ok, {wave_failed} failed in {elapsed:.1f}s", flush=True)
    
    return wave_idx, wave_success, wave_failed, failed_batches


async def main_async():
    """Main function to process Reddit data"""
    print("=" * 80)
    print("YOUTUBE Data Labeling Script")
    print("=" * 80)
    
    if not DCOMPUTE_API_KEY:
        print("ERROR: Missing DCOMPUTE_API_KEY (or OPENAI_API_KEY) in environment")
        return

    print(f"Using model: {MODEL_NAME}")
    print(f"Endpoint: {DCOMPUTE_BASE_URL}")
    client = AsyncOpenAI(base_url=DCOMPUTE_BASE_URL, api_key=DCOMPUTE_API_KEY)
    
    guidelines = load_annotation_guidelines()
    
    # Load progress
    # Load progress
    progress = load_progress()
    completed_waves = progress.get('completed_waves', set())

    # Auto-detect completed waves from disk (handles fresh progress file + existing wave files)
    if os.path.exists(WAVES_DIR):
        for fname in os.listdir(WAVES_DIR):
            if fname.startswith("wave_") and fname.endswith(".jsonl"):
                try:
                    wave_idx = int(fname.replace("wave_", "").replace(".jsonl", ""))
                    fpath = os.path.join(WAVES_DIR, fname)
                    if os.path.getsize(fpath) > 0:  # only count non-empty files
                        completed_waves.add(wave_idx)
                except ValueError:
                    pass
        if completed_waves:
            print(f"  Auto-detected {len(completed_waves)} completed waves from disk")
            save_progress(completed_waves, 0)  # sync progress file to disk state

    if not BATCHES_DIR:
        raise ValueError(
        "YOUTUBE_BATCHES_DIR not set.\n"
        "Run:\n"
        "python prepare_youtube_batches.py\n"
        )

    print(f"\nLoading pre-split batches from {BATCHES_DIR}...")
    all_batches = load_batches_from_manifest(BATCHES_DIR)
    total_rows = all_batches[-1][1] if all_batches else 0
    print(f"Detected {len(all_batches)} total batch files")
    
    # Pre-split into independent waves
    NUM_WAVES = min(4000, len(all_batches))
    batches_per_wave = (len(all_batches) + NUM_WAVES - 1) // NUM_WAVES
    waves = [
        all_batches[i:i+batches_per_wave]
        for i in range(0, len(all_batches), batches_per_wave)
    ]
    
    if completed_waves:
        print(f"\nResuming: {len(completed_waves)} of {len(waves)} waves already done")

    # Create tasks for incomplete waves only, with disk-level validation
    wave_tasks = {}
    for wave_idx, wave_batches in enumerate(waves):
        wave_file = os.path.join(WAVES_DIR, f"wave_{wave_idx:02d}.jsonl")
        already_done = wave_idx in completed_waves and os.path.exists(wave_file)

        if already_done:
            print(f"  Skipping wave {wave_idx} (already complete on disk)")
        else:
            if wave_idx in completed_waves:
                # Progress JSON says done but wave file is missing — re-run it
                print(f"  Wave {wave_idx} marked done in progress file but wave file missing — re-running")
                completed_waves.discard(wave_idx)
            wave_tasks[wave_idx] = asyncio.create_task(
                process_wave(wave_idx, wave_batches, client, guidelines)
            )

    print(f"\nProcessing {total_rows} rows in {len(waves)} independent parallel waves")
    print(f"Waves to run this session: {len(wave_tasks)} | Already done: {len(waves) - len(wave_tasks)}")
    print("-" * 80)

    all_failed = []
    if wave_tasks:
        results = await asyncio.gather(*wave_tasks.values(), return_exceptions=False)
        
        for result in results:
            wave_idx, wave_success, wave_failed, wave_failed_batches = result
            all_failed.extend(wave_failed_batches)

            # Mark wave complete and save progress immediately after each wave
            completed_waves.add(wave_idx)
            save_progress(completed_waves, len(waves))

            wave_batches = waves[wave_idx]
            wave_rows_start = wave_batches[0][0]
            wave_rows_end = wave_batches[-1][1] - 1
            progress_pct = (len(completed_waves) / len(waves)) * 100
            print(
                f"Wave {wave_idx + 1}/{len(waves)} | rows {wave_rows_start}-{wave_rows_end} | "
                f"ok={wave_success}, failed={wave_failed} | "
                f"progress={len(completed_waves)}/{len(waves)} ({progress_pct:.1f}%)"
            )

    # Final output
    print("\n" + "=" * 80)
    if len(completed_waves) == len(waves):
        print("✓ Processing complete! All waves done.")
        merge_wave_files(len(waves))
    else:
        print(f"Processing paused. Completed {len(completed_waves)}/{len(waves)} waves.")
        print(f"Partial results in {WAVES_DIR}/")

    save_progress(completed_waves, len(waves))
    print(f"Total waves: {len(waves)}")
    print(f"Completed waves: {len(completed_waves)}")
    if all_failed:
        print(f"Failed batches: {len(all_failed)}")
        for batch_start, batch_end, error_msg in all_failed[:10]:
            print(f"  - Rows {batch_start} to {batch_end - 1}: {error_msg}")
        if len(all_failed) > 10:
            print(f"  ... and {len(all_failed) - 10} more")
    print("=" * 80)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()