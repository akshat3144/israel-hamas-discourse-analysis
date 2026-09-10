# -*- coding: utf-8 -*-
"""
Stage 6b: score the human validation (reviewer R7).

Run AFTER at least two annotators have filled the `human_label` column in their
`06_revision/validation/annotator_*.csv`.

Reports:
  * inter-annotator agreement (pairwise Cohen's kappa, Fleiss' kappa,
    Krippendorff's alpha for nominal data)
  * the human gold standard (majority vote across annotators)
  * LLM-vs-human accuracy and macro-F1, overall / per platform / per class
  * High vs Medium-Low accuracy -> does the confidence filter earn its place?
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)

HERE = Path(__file__).resolve().parent
VAL = HERE / "validation"
OUT = HERE / "outputs"
LABELS = ["P", "I", "N", "R"]


def fleiss_kappa(table):
    """table: n_items x n_categories counts."""
    n_items, n_cat = table.shape
    n_ann = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / (n_items * n_ann)
    P_i = ((table ** 2).sum(axis=1) - n_ann) / (n_ann * (n_ann - 1))
    P_bar = P_i.mean()
    Pe = (p_j ** 2).sum()
    return (P_bar - Pe) / (1 - Pe) if (1 - Pe) else np.nan


def krippendorff_alpha_nominal(matrix):
    """matrix: n_annotators x n_items, np.nan for missing."""
    vals, counts = {}, 0
    units = []
    for j in range(matrix.shape[1]):
        col = [v for v in matrix[:, j] if isinstance(v, str)]
        if len(col) >= 2:
            units.append(col)
    if not units:
        return np.nan
    # observed disagreement
    Do_num, Do_den = 0.0, 0.0
    allvals = []
    for u in units:
        m = len(u)
        allvals.extend(u)
        for a, b in itertools.permutations(u, 2):
            Do_num += (a != b)
        Do_den += (m - 1)
    Do = Do_num / (len(units) * 0 + sum(len(u) * (len(u) - 1) for u in units))
    # expected disagreement
    from collections import Counter
    c = Counter(allvals)
    n = sum(c.values())
    De = 1 - sum(v * (v - 1) for v in c.values()) / (n * (n - 1))
    return 1 - Do / De if De else np.nan


def main():
    key = pd.read_csv(VAL / "validation_key.csv")
    files = sorted(VAL.glob("annotator_*.csv"))
    ann = {}
    for f in files:
        d = pd.read_csv(f)
        if "human_label" not in d.columns:
            continue
        d["human_label"] = d["human_label"].astype(str).str.strip().str.upper()
        d = d[d["human_label"].isin(LABELS)]
        if len(d) > 0:
            ann[f.stem[-1]] = d.set_index("item_id")["human_label"]
            print(f"  {f.name}: {len(d)} labelled")
    if len(ann) < 2:
        print("\nNeed at least TWO annotators with a filled `human_label` column.")
        print("Fill 06_revision/validation/annotator_A.csv and annotator_B.csv, then re-run.")
        return

    A = pd.DataFrame(ann)
    A = A.dropna(how="all")
    print(f"\nItems with >=1 human label: {len(A)}")

    res = {"n_annotators": len(ann), "n_items": int(len(A))}

    # ---- inter-annotator agreement
    print("\n=== Inter-annotator agreement ===")
    pk = {}
    for a, b in itertools.combinations(A.columns, 2):
        both = A[[a, b]].dropna()
        k = cohen_kappa_score(both[a], both[b])
        pk[f"{a}-{b}"] = round(float(k), 4)
        print(f"  Cohen's kappa {a}-{b}: {k:.4f}  (n={len(both)})")
    res["pairwise_cohen_kappa"] = pk

    complete = A.dropna()
    if len(complete) and len(A.columns) >= 2:
        tab = np.zeros((len(complete), len(LABELS)), dtype=int)
        for i, (_, row) in enumerate(complete.iterrows()):
            for v in row:
                tab[i, LABELS.index(v)] += 1
        fk = fleiss_kappa(tab)
        res["fleiss_kappa"] = round(float(fk), 4)
        print(f"  Fleiss' kappa (n={len(complete)} complete items): {fk:.4f}")
    ka = krippendorff_alpha_nominal(A.to_numpy().T)
    res["krippendorff_alpha"] = round(float(ka), 4)
    print(f"  Krippendorff's alpha (nominal): {ka:.4f}")

    # ---- human gold standard = majority vote (ties -> first annotator)
    def majority(row):
        vals = [v for v in row if isinstance(v, str)]
        if not vals:
            return np.nan
        from collections import Counter
        c = Counter(vals).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            return vals[0]
        return c[0][0]

    gold = A.apply(majority, axis=1).rename("human_gold")
    merged = key.set_index("item_id").join(gold, how="inner").dropna(subset=["human_gold"])
    print(f"\nGold-standard items: {len(merged)}")

    # ---- LLM vs human
    print("\n=== LLM vs human ===")
    acc = accuracy_score(merged["human_gold"], merged["model_label"])
    f1 = f1_score(merged["human_gold"], merged["model_label"], average="macro")
    res["overall"] = {"accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
                      "n": int(len(merged))}
    print(f"  overall accuracy={acc:.4f}  macro-F1={f1:.4f}  (n={len(merged)})")
    print("\n" + classification_report(merged["human_gold"], merged["model_label"], zero_division=0))

    labs = sorted(set(merged["human_gold"]) | set(merged["model_label"]))
    cm = confusion_matrix(merged["human_gold"], merged["model_label"], labels=labs)
    print("Confusion (rows=human gold, cols=model):")
    print(pd.DataFrame(cm, index=labs, columns=labs).to_string())
    res["confusion"] = {"labels": labs, "matrix": cm.tolist()}

    # ---- per platform
    res["by_platform"] = {}
    for p, sub in merged.groupby("platform"):
        a = accuracy_score(sub["human_gold"], sub["model_label"])
        res["by_platform"][p] = {"accuracy": round(float(a), 4), "n": int(len(sub))}
        print(f"  {p}: accuracy={a:.4f} (n={len(sub)})")

    # ---- KEY: does the confidence filter earn its place?
    print("\n=== Confidence filter validation (R7) ===")
    res["by_confidence"] = {}
    for c, sub in merged.groupby("confidence"):
        a = accuracy_score(sub["human_gold"], sub["model_label"])
        res["by_confidence"][c] = {"accuracy": round(float(a), 4), "n": int(len(sub))}
        print(f"  {c}: accuracy={a:.4f} (n={len(sub)})")
    if {"High", "LowMed"} <= set(res["by_confidence"]):
        hi = res["by_confidence"]["High"]["accuracy"]
        lo = res["by_confidence"]["LowMed"]["accuracy"]
        print(f"  -> High minus Medium/Low = {hi - lo:+.4f}")
        print("     (a clear positive gap empirically justifies keeping only High-confidence labels)")
        res["confidence_gap"] = round(float(hi - lo), 4)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "human_validation.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {OUT / 'human_validation.json'}")


if __name__ == "__main__":
    main()
