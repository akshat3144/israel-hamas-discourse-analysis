# -*- coding: utf-8 -*-
"""
Stage 6: build the human-validation annotation package (reviewer R7).

R7: "Automated annotation must always have a human validation score."

This produces a blind annotation sheet plus a hidden answer key, sampled so that
it validates BOTH:
  (a) the accuracy of the labels we actually keep  (High confidence, P/I/N), and
  (b) the confidence filter itself, by including Medium/Low-confidence items that
      we discarded - if High really is more accurate, the filter is justified.

Outputs (06_revision/validation/):
  validation_blind.csv     -> give this to annotators (NO model label shown)
  validation_key.csv       -> hidden key with the model label + confidence
  annotator_template.csv   -> one per annotator, to fill in a `human_label` column
"""
import json
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "00_data_collection_and_labeling" / "outputs"
VAL = HERE / "validation"
VAL.mkdir(parents=True, exist_ok=True)

SEED = 42
N_HIGH = 30      # per stance, per platform  (validates retained labels)
N_LOWMED = 15    # per stance, per platform  (validates the confidence filter)
STANCES = ["P", "I", "N"]
random.seed(SEED)


def conf_tier(c):
    c = str(c).strip().lower()
    return "High" if c in {"high", "very high", "very_high", "high confidence", "strong"} else "LowMed"


def reservoir_sample(path, platform, text_field, ctx_fields, cap_per_cell):
    """Single streaming pass with per-cell reservoir sampling."""
    cells = {(s, t): [] for s in STANCES for t in ("High", "LowMed")}
    seen = {k: 0 for k in cells}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            lab = o.get("Label") or o.get("label")
            if lab not in STANCES:
                continue
            tier = conf_tier(o.get("Confidence") or o.get("confidence"))
            key = (lab, tier)
            k = cap_per_cell[tier]
            seen[key] += 1
            rec = {
                "platform": platform,
                "row_index": o.get("index"),
                "text": str(o.get(text_field, ""))[:1500],
                "model_label": lab,
                "confidence": tier,
                "model_reasoning": str(o.get("Reasoning", ""))[:500],
            }
            for cf, alias in ctx_fields.items():
                rec[alias] = str(o.get(cf, ""))[:300]
            buf = cells[key]
            if len(buf) < k:
                buf.append(rec)
            else:
                j = random.randint(0, seen[key] - 1)
                if j < k:
                    buf[j] = rec
    return cells, seen


cap = {"High": N_HIGH, "LowMed": N_LOWMED}

print("Streaming Reddit labeled data (2.9 GB, one pass) ...")
r_cells, r_seen = reservoir_sample(
    RAW / "reddit_labeled_full.jsonl", "reddit", "self_text",
    {"subreddit": "context_subreddit", "post_title": "context_post_title"}, cap)

print("Streaming YouTube labeled data ...")
y_cells, y_seen = reservoir_sample(
    RAW / "youtube_labeled_full.jsonl", "youtube", "text",
    {"video id": "context_video_id"}, cap)

rows = []
for cells in (r_cells, y_cells):
    for key, buf in cells.items():
        rows.extend(buf)
df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "item_id", [f"V{i:04d}" for i in range(1, len(df) + 1)])

print(f"\nSampled {len(df)} items")
print(df.groupby(["platform", "model_label", "confidence"]).size().to_string())

# Blind sheet: everything the annotator needs, nothing that reveals the model
blind_cols = ["item_id", "platform", "text", "context_subreddit",
              "context_post_title", "context_video_id"]
blind = df.reindex(columns=blind_cols).fillna("")
blind.to_csv(VAL / "validation_blind.csv", index=False, encoding="utf-8-sig")

# Hidden key
key = df[["item_id", "platform", "row_index", "model_label", "confidence", "model_reasoning"]]
key.to_csv(VAL / "validation_key.csv", index=False, encoding="utf-8-sig")

# Annotator templates
tmpl = blind.copy()
tmpl["human_label"] = ""      # annotator fills: P / I / N / R
tmpl["notes"] = ""
for a in ("A", "B", "C"):
    tmpl.to_csv(VAL / f"annotator_{a}.csv", index=False, encoding="utf-8-sig")

readme = f"""# Human validation package (reviewer point R7)

{len(df)} items sampled from the *pre-filter* labelled data so that we can
validate two things at once:

1. **Accuracy of the labels we keep** - the High-confidence P/I/N items
   ({N_HIGH} per stance per platform).
2. **Whether the confidence filter is justified** - the Medium/Low-confidence
   items we discarded ({N_LOWMED} per stance per platform). If accuracy is
   materially higher in the High tier, the filter is empirically justified
   rather than assumed.

## How to annotate
Each annotator opens their own `annotator_X.csv` and fills the `human_label`
column with exactly one of:

  P = Supports Palestine   I = Supports Israel
  N = Neutral / unclear    R = Irrelevant to the conflict

Use `00_data_collection_and_labeling/ANNOTATION_GUIDELINES.md` as the rulebook.
Annotate independently and do not discuss items while labelling - the whole point
is to measure independent agreement.

**Do not open `validation_key.csv`** until all annotators have finished; it
contains the model's labels.

## Scoring
Once at least two annotators are done:

    python 06_revision/score_validation.py

which reports inter-annotator agreement (Krippendorff's alpha / Cohen's kappa),
model-vs-human accuracy and macro-F1, a per-class confusion matrix, and the
High-vs-Medium/Low accuracy comparison.
"""
(VAL / "README.md").write_text(readme, encoding="utf-8")

print(f"\nWrote -> {VAL}")
for p in sorted(VAL.iterdir()):
    print("   ", p.name)
