# -*- coding: utf-8 -*-
"""
Controlled test of what video context is worth to the YouTube annotator.

Two arms, identical model / provider / prompt scaffolding / temperature.
The ONLY difference is whether each item carries the video title and
description. Both arms are scored against the same three-annotator human
gold standard.

  Arm A: replicates the original YouTube prompt exactly (video ID only)
  Arm B: same, plus Video Title and Video Description

Arm A doubles as a reproduction check against the corpus labels (0.618).
"""
import glob
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(r"C:\Users\nitro 5\Desktop\Projects\israel_hamas_discourse_analysis")
VAL = ROOT / "06_revision" / "validation"
META = Path(r"C:\Users\nitro 5\Desktop\youtube_video_metadata.csv")
OUTDIR = Path(__file__).resolve().parent
MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.1
BATCH = 10
DESC_CHARS = 600

# --- pull the ORIGINAL guidelines text verbatim from the labelling script ---
spec = importlib.util.spec_from_file_location(
    "lyt", ROOT / "00_data_collection_and_labeling" / "scripts" / "label_youtube.py")
lyt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lyt)
GUIDELINES = lyt.load_annotation_guidelines()


def human_gold():
    key = pd.read_csv(VAL / "validation_key.csv").set_index("item_id")
    blind = pd.read_csv(VAL / "validation_blind.csv").set_index("item_id")
    ann = {}
    for f in sorted(glob.glob(str(VAL / "annotator_*.xlsx"))):
        who = os.path.basename(f).split("annotator_", 1)[-1].replace(".xlsx", "")
        d = pd.read_excel(f, sheet_name="Annotate", header=3)
        ann[who] = d.assign(
            l=d.human_label.astype(str).str.strip().str.upper()
        ).set_index("item_id")["l"]
    A = pd.DataFrame(ann)

    def maj(r):
        c = Counter(r).most_common()
        return None if c[0][1] == 1 else c[0][0]

    df = key.join(A.apply(maj, axis=1).rename("gold")).join(
        blind[["text", "context_video_id"]])
    df = df[(df.platform == "youtube") & df.gold.notna()]
    return df


def build_prompt(records, with_context):
    lines = [f"""You are an expert annotator analyzing discourse about the Israel-Hamas war on YouTube.

{GUIDELINES}

Analyze the following {len(records)} YouTube comments.

For each item consider:
- Comment text
- Video context
- Author
- Metadata if available

Data:
"""]
    for r in records:
        block = [f"\n--- Item {r['idx']} ---\n",
                 f"Comment ID: {r['comment_id']}",
                 f"Video ID: {r['video_id']}"]
        if with_context:
            block.append(f"Video Title: {r['title']}")
            block.append(f"Video Description: {r['description']}")
        block += [f"Author: {r['author']}",
                  f"Likes: {r['likes']}",
                  "", "Comment:", r["comment"], ""]
        lines.append("\n".join(block))

    lines.append(f"""

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
""")
    return "\n".join(lines)


def parse(text):
    t = text.strip()
    cands = []
    if "```json" in t:
        cands.append(t.split("```json", 1)[1].split("```", 1)[0])
    if "```" in t:
        cands.append(t.split("```", 1)[1].split("```", 1)[0])
    a, b = t.find("["), t.rfind("]")
    if a != -1 and b > a:
        cands.append(t[a:b + 1])
    cands.append(t)
    for c in cands:
        for attempt in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                p = json.loads(attempt, strict=False)
                if isinstance(p, list):
                    return p
            except Exception:
                pass
    return None


def run_arm(client, records, with_context, tag):
    got = {}
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        prompt = build_prompt(chunk, with_context)
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=TEMPERATURE, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}])
                out = parse(r.choices[0].message.content or "")
                if out is None:
                    raise ValueError("unparseable")
                # the model emits 0-based positions within the batch, not our
                # item ids, so map by position and sanity-check the count
                if len(out) != len(chunk):
                    raise ValueError(f"expected {len(chunk)} objects, got {len(out)}")
                for pos, o in enumerate(out):
                    lab = str(o.get("Label", "")).strip().upper()
                    if lab in ("P", "I", "N", "R"):
                        got[chunk[pos]["item_id"]] = lab
                break
            except Exception as e:
                if attempt == 4:
                    print(f"    batch {i} failed: {str(e)[:90]}")
                else:
                    time.sleep(3 * (attempt + 1))
        print(f"  [{tag}] {min(i+BATCH,len(records))}/{len(records)}", flush=True)
        time.sleep(1.5)
    return got


def main():
    g = human_gold()
    meta = pd.read_csv(META).drop_duplicates("video_id").set_index("video_id")
    print(f"YouTube validation items with a human majority: {len(g)}")

    records = []
    for iid, r in g.iterrows():
        vid = str(r.context_video_id)
        m = meta.loc[vid] if vid in meta.index else None
        records.append({
            "item_id": iid,
            "idx": int(r.row_index),
            "comment_id": "N/A",
            "video_id": vid,
            "author": "N/A",
            "likes": "N/A",
            "comment": str(r.text)[:1200],
            "title": str(m.title) if m is not None else "N/A",
            "description": (str(m.description)[:DESC_CHARS]
                            if m is not None and pd.notna(m.description) else "N/A"),
        })
    have = sum(1 for r in records if r["title"] != "N/A")
    print(f"items with video metadata: {have}/{len(records)}\n")

    client = OpenAI(base_url="https://api.groq.com/openai/v1",
                    api_key=os.environ["GROQ_KEY"], timeout=120)

    print("ARM A - no video context (replicates the original prompt)")
    a = run_arm(client, records, False, "A")
    print("\nARM B - with video title and description")
    b = run_arm(client, records, True, "B")

    rows = []
    for r in records:
        rows.append({"item_id": r["item_id"], "idx": r["idx"],
                     "gold": g.loc[r["item_id"], "gold"],
                     "corpus": g.loc[r["item_id"], "model_label"],
                     "confidence": g.loc[r["item_id"], "confidence"],
                     "armA": a.get(r["item_id"]), "armB": b.get(r["item_id"])})
    d = pd.DataFrame(rows)
    d.to_csv(OUTDIR / "context_experiment_raw.csv", index=False)

    def acc(col, sub=None):
        x = d if sub is None else d[sub]
        x = x[x[col].notna()]
        return (x[col] == x.gold).mean(), len(x)

    print("\n" + "=" * 62)
    print("scored against the three-annotator human gold standard")
    print("=" * 62)
    for name, col in (("corpus labels (dcompute, Llama-3.3)", "corpus"),
                      ("Arm A  gpt-oss-120b, NO context", "armA"),
                      ("Arm B  gpt-oss-120b, WITH context", "armB")):
        v, n = acc(col)
        print(f"  {name:<38} {v:.3f}  (n={n})")

    hi = d.confidence == "High"
    pin = d.gold.isin(["P", "I", "N"])
    print("\n  restricted to High-confidence, stance-gold items:")
    for name, col in (("corpus", "corpus"), ("Arm A", "armA"), ("Arm B", "armB")):
        v, n = acc(col, hi & pin)
        print(f"    {name:<8} {v:.3f}  (n={n})")

    va, _ = acc("armA"); vb, _ = acc("armB")
    print(f"\n  effect of adding video context: {vb - va:+.3f} "
          f"({100*(vb-va):+.1f} points)")
    json.dump({"arm_a": va, "arm_b": vb, "delta": vb - va},
              open(OUTDIR / "context_experiment.json", "w"), indent=2)
    print(f"\nraw per-item results -> {OUTDIR / 'context_experiment_raw.csv'}")


if __name__ == "__main__":
    main()
