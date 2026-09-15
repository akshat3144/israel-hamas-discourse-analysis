# -*- coding: utf-8 -*-
"""
Per-platform detail behind human_validation.json.

The headline scorer reports pooled numbers. The paper's claims are all
cross-platform comparisons, so what matters is whether label quality differs
BY PLATFORM - if it does, every stance-conditional contrast carries unequal
attenuation and has to be defended.

Run:  python 06_revision/validation_breakdown.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

HERE = Path(__file__).resolve().parent
VAL = HERE / "validation"
OUT = HERE / "outputs"
LABELS = ["P", "I", "N", "R"]


def majority(row):
    vals = [v for v in row if isinstance(v, str)]
    if not vals:
        return None
    c = Counter(vals).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return vals[0]
    return c[0][0]


def main():
    key = pd.read_csv(VAL / "validation_key.csv").set_index("item_id")
    ann = {}
    for f in sorted(VAL.glob("annotator_*.xlsx")):
        who = f.stem.split("annotator_", 1)[-1]
        d = pd.read_excel(f, sheet_name="Annotate", header=3)
        d["human_label"] = d["human_label"].astype(str).str.strip().str.upper()
        ann[who] = d.set_index("item_id")["human_label"]
    A = pd.DataFrame(ann)
    gold = A.apply(majority, axis=1).rename("human_gold")
    df = key.join(gold).dropna(subset=["human_gold"])

    res = {"n": int(len(df)), "annotators": sorted(ann)}
    print(f"{len(df)} items, {len(ann)} annotators\n")

    # unanimity: how solid is the gold standard itself?
    unan = A.apply(lambda r: len(set(r.dropna())) == 1, axis=1)
    res["unanimous_pct"] = round(100 * float(unan.mean()), 1)
    print(f"all three agree on {100*unan.mean():.1f}% of items")
    df["unanimous"] = unan.reindex(df.index).values
    u = df[df["unanimous"]]
    print(f"  model accuracy on unanimous items: "
          f"{accuracy_score(u['human_gold'], u['model_label']):.3f} (n={len(u)})")
    res["acc_on_unanimous"] = round(
        float(accuracy_score(u["human_gold"], u["model_label"])), 4)

    print("\n=== by platform ===")
    res["by_platform"] = {}
    for p, sub in df.groupby("platform"):
        pin = sub[sub["human_gold"].isin(["P", "I", "N"])]
        r_rate = 100.0 * (sub["human_gold"] == "R").mean()
        row = {
            "n": int(len(sub)),
            "acc_4class": round(float(accuracy_score(sub["human_gold"], sub["model_label"])), 4),
            "n_pin": int(len(pin)),
            "acc_pin": round(float(accuracy_score(pin["human_gold"], pin["model_label"])), 4),
            "macro_f1_pin": round(float(f1_score(pin["human_gold"], pin["model_label"],
                                                 average="macro", zero_division=0)), 4),
            "irrelevant_pct": round(float(r_rate), 1),
        }
        # inter-annotator agreement within this platform
        ids = sub.index
        ks = []
        names = sorted(ann)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = A.loc[ids, names[i]], A.loc[ids, names[j]]
                ks.append(cohen_kappa_score(a, b))
        row["human_kappa_mean"] = round(float(sum(ks) / len(ks)), 4)
        res["by_platform"][p] = row
        print(f"  {p:<8} n={row['n']:<4} human-kappa={row['human_kappa_mean']:.3f}  "
              f"model acc(P/I/N)={row['acc_pin']:.3f}  macro-F1={row['macro_f1_pin']:.3f}  "
              f"irrelevant={row['irrelevant_pct']:.1f}%")

    print("\n=== irrelevance by platform x confidence ===")
    res["irrelevant"] = {}
    for (p, c), sub in df.groupby(["platform", "confidence"]):
        pct = round(100.0 * float((sub["human_gold"] == "R").mean()), 1)
        res["irrelevant"][f"{p}/{c}"] = pct
        print(f"  {p:<8} {c:<7} {pct:>5.1f}%  (n={len(sub)})")

    print("\n=== High-tier only, the corpus we actually analyse ===")
    res["high_tier"] = {}
    hi = df[df["confidence"] == "High"]
    for p, sub in hi.groupby("platform"):
        pin = sub[sub["human_gold"].isin(["P", "I", "N"])]
        d = {
            "n": int(len(sub)),
            "irrelevant_pct": round(100.0 * float((sub["human_gold"] == "R").mean()), 1),
            "acc_pin": round(float(accuracy_score(pin["human_gold"], pin["model_label"])), 4),
            "n_pin": int(len(pin)),
        }
        res["high_tier"][p] = d
        print(f"  {p:<8} n={d['n']:<4} irrelevant={d['irrelevant_pct']:>5.1f}%  "
              f"stance acc on the rest={d['acc_pin']:.3f} (n={d['n_pin']})")

    print("\n=== where the model goes wrong (High tier, P/I/N gold) ===")
    hp = hi[hi["human_gold"].isin(["P", "I", "N"])]
    cm = pd.crosstab(hp["human_gold"], hp["model_label"])
    print(cm.to_string())
    res["high_tier_confusion"] = cm.to_dict()

    dest = OUT / "validation_breakdown.json"
    dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
