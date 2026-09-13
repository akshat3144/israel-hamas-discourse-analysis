# -*- coding: utf-8 -*-
"""
Annotation reliability metrics that do not require the human study (reviewer R7
follow-up: "highlight the accuracy of your methods in the paper").

Three machine-checkable properties of the LLM annotation are measured on the
*pre-filter* labelled data:

1. Confidence-tier composition - how much of the corpus the High filter keeps.
2. Test-retest self-consistency - identical comment texts that were submitted to
   the annotator in *different* batches (>= 50 rows apart, the batch size) are
   independent re-annotations of the same item. Agreement on those pairs is a
   genuine intra-annotator reliability coefficient. Reported twice: over all
   repeated texts (a lower bound, since the surrounding post/video context may
   differ and legitimately change the label) and over repeats within the *same*
   post/video, where the annotator saw identical input both times.
3. Confidence-filter validity - a supervised model trained on High-tier labels
   recovers held-out High-tier labels better than Medium/Low-tier ones, which
   makes the filter an empirical choice rather than an assumption.

Usage:  python 06_revision/annotation_reliability.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def find_repo_root():
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "reddit_labeled.csv").exists():
            return candidate
    return here


ROOT = find_repo_root()
LABELED = ROOT / "00_data_collection_and_labeling" / "outputs"
OUT = ROOT / "06_revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "reddit": (LABELED / "reddit_labeled_full.jsonl", "self_text", "post_id"),
    "youtube": (LABELED / "youtube_labeled_full.jsonl", "text", "video id"),
}

LAB2CODE = {"P": 0, "I": 1, "N": 2, "R": 3}
CODE2LAB = {v: k for k, v in LAB2CODE.items()}
CONF2CODE = {"High": 0, "Medium": 1, "Low": 2}

BATCH = 50          # labelling batch size; duplicates must be further apart
MIN_CHARS = 20      # ignore "lol" / "yes" style stubs that duplicate by chance
MAX_CHARS = 2000
KEEP_HIGH = 0.04    # subsample rates for the supervised tier check
KEEP_LOW = 0.35
CAP = 150_000


def normalise(s):
    return " ".join(s.lower().split())


def stable_hash(s):
    """Reproducible 64-bit key (Python's hash() is salted per process)."""
    d = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(d, "big", signed=True)


def scan(path, text_field, ctx_field, rng):
    """One streaming pass: tier counts, duplicate keys, and a text subsample."""
    hashes, chashes, codes, confs, idxs = [], [], [], [], []
    tier_counts = Counter()
    sample = {"High": [], "Low": []}
    n_lines = 0

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lab = row.get("Label")
            conf = row.get("Confidence")
            if lab not in LAB2CODE or conf not in CONF2CODE:
                continue
            tier_counts[(conf, lab)] += 1

            text = row.get(text_field) or ""
            if not isinstance(text, str):
                continue
            norm = normalise(text)
            if MIN_CHARS <= len(norm) <= MAX_CHARS:
                ctx = str(row.get(ctx_field) or "")
                hashes.append(stable_hash(norm))
                chashes.append(stable_hash(norm + "" + ctx))
                codes.append(LAB2CODE[lab])
                confs.append(CONF2CODE[conf])
                idxs.append(int(row.get("index", n_lines)))

                if lab != "R":          # tier check uses the analytic classes
                    bucket = "High" if conf == "High" else "Low"
                    rate = KEEP_HIGH if bucket == "High" else KEEP_LOW
                    if len(sample[bucket]) < CAP and rng.random() < rate:
                        sample[bucket].append((norm, LAB2CODE[lab]))

            if n_lines % 500_000 == 0:
                print(f"    {n_lines:,} lines", flush=True)

    return (np.array(hashes, dtype=np.int64),
            np.array(chashes, dtype=np.int64),
            np.array(codes, dtype=np.int8),
            np.array(confs, dtype=np.int8),
            np.array(idxs, dtype=np.int64),
            tier_counts, sample, n_lines)


def cohen_kappa(a, b, k=4):
    m = np.zeros((k, k), dtype=float)
    np.add.at(m, (a, b), 1.0)
    m /= m.sum()
    po = float(np.trace(m))
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def duplicate_consistency(hashes, codes, confs, idxs):
    """Agreement between independent re-annotations of identical text."""
    order = np.argsort(hashes, kind="mergesort")
    h, c, cf, ix = hashes[order], codes[order], confs[order], idxs[order]

    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    ends = np.r_[starts[1:], len(h)]

    first, second = [], []
    first_hi, second_hi = [], []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        gi, gc, gf = ix[s:e], c[s:e], cf[s:e]
        o = np.argsort(gi)
        gi, gc, gf = gi[o], gc[o], gf[o]
        # earliest pair that straddles a batch boundary
        far = np.flatnonzero(gi - gi[0] >= BATCH)
        if far.size == 0:
            continue
        j = far[0]
        first.append(gc[0])
        second.append(gc[j])
        if gf[0] == 0 and gf[j] == 0:
            first_hi.append(gc[0])
            second_hi.append(gc[j])

    out = {"n_pairs": len(first)}
    if first:
        k, po = cohen_kappa(np.array(first), np.array(second))
        out["agreement_pct"] = round(100 * po, 2)
        out["kappa"] = round(k, 4)
    if first_hi:
        k, po = cohen_kappa(np.array(first_hi), np.array(second_hi))
        out["n_pairs_high"] = len(first_hi)
        out["agreement_pct_high"] = round(100 * po, 2)
        out["kappa_high"] = round(k, 4)
    return out


def tier_validity(sample, seed=42):
    """Train on High-tier labels; compare recovery on High vs Medium/Low."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    hi, lo = sample["High"], sample["Low"]
    if len(hi) < 5000 or len(lo) < 1000:
        return {"error": "insufficient sample"}

    X_hi = [t for t, _ in hi]
    y_hi = np.array([c for _, c in hi])
    X_lo = [t for t, _ in lo]
    y_lo = np.array([c for _, c in lo])

    Xtr, Xte, ytr, yte = train_test_split(
        X_hi, y_hi, test_size=0.2, random_state=seed, stratify=y_hi)

    vec = TfidfVectorizer(max_features=60_000, ngram_range=(1, 2),
                          min_df=3, sublinear_tf=True)
    Ztr = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    clf.fit(Ztr, ytr)

    p_hi = clf.predict(vec.transform(Xte))
    p_lo = clf.predict(vec.transform(X_lo))
    return {
        "n_train_high": len(Xtr),
        "n_test_high": len(Xte),
        "n_test_medlow": len(X_lo),
        "acc_high": round(float(accuracy_score(yte, p_hi)), 4),
        "acc_medlow": round(float(accuracy_score(y_lo, p_lo)), 4),
        "macro_f1_high": round(float(f1_score(yte, p_hi, average="macro")), 4),
        "macro_f1_medlow": round(float(f1_score(y_lo, p_lo, average="macro")), 4),
    }


def main():
    rng = np.random.default_rng(20240507)
    results = {}

    for platform, (path, field, ctx) in SOURCES.items():
        if not path.exists():
            print(f"  !! missing {path}")
            continue
        print(f"[{platform}] scanning {path.name} ...", flush=True)
        h, ch, c, cf, ix, tiers, sample, n_lines = scan(path, field, ctx, rng)

        total = sum(tiers.values())
        by_conf = Counter()
        for (conf, lab), n in tiers.items():
            by_conf[conf] += n
        high_non_r = sum(n for (conf, lab), n in tiers.items()
                         if conf == "High" and lab != "R")

        print(f"  lines={n_lines:,} usable={total:,} texts={len(h):,}", flush=True)
        print("  duplicate consistency ...", flush=True)
        dup = duplicate_consistency(h, c, cf, ix)
        print(f"    any-context: {dup}", flush=True)
        dup_ctx = duplicate_consistency(ch, c, cf, ix)
        print(f"    same-context: {dup_ctx}", flush=True)

        print("  confidence-tier validity ...", flush=True)
        tv = tier_validity(sample)
        print(f"    {tv}", flush=True)

        results[platform] = {
            "n_lines": n_lines,
            "n_labelled": total,
            "tier_pct": {k: round(100 * v / total, 2) for k, v in by_conf.items()},
            "high_non_irrelevant": high_non_r,
            "tier_by_label_pct": {
                f"{conf}/{lab}": round(100 * n / total, 3)
                for (conf, lab), n in sorted(tiers.items())
            },
            "duplicate_consistency": dup,
            "duplicate_consistency_same_context": dup_ctx,
            "confidence_tier_validity": tv,
        }

    dest = OUT / "annotation_reliability.json"
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
