# -*- coding: utf-8 -*-
"""Audit citations, figure/table refs, and numeric consistency against the
analysis outputs."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "06_revision" / "outputs"

# Concatenate all source the paper actually compiles
src = ""
for f in ["body_new.tex", "rw_platform.tex", "rw_conflict.tex", "rq2_body.tex"]:
    src += (HERE / f).read_text(encoding="utf-8") + "\n"

bib = (HERE / "refs.bib").read_text(encoding="utf-8")

print("=" * 62)
print("CITATION AUDIT")
print("=" * 62)
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for m in re.findall(r"\\cite\{([^}]*)\}", src):
    cited.update(k.strip() for k in m.split(","))
print(f"  defined in refs.bib : {len(defined)}")
print(f"  cited in paper      : {len(cited)}")
missing = cited - defined
orphan = defined - cited
print(f"  cited but UNDEFINED : {sorted(missing) if missing else 'none'}")
print(f"  defined but UNUSED  : {sorted(orphan) if orphan else 'none'}")

print("\n" + "=" * 62)
print("CROSS-REFERENCE AUDIT")
print("=" * 62)
labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
print(f"  labels: {len(labels)}   refs: {len(refs)}")
print(f"  ref'd but no label  : {sorted(refs - labels) if refs - labels else 'none'}")
print(f"  label never ref'd   : {sorted(labels - refs) if labels - refs else 'none'}")

print("\n" + "=" * 62)
print("NUMERIC CONSISTENCY (paper text vs analysis outputs)")
print("=" * 62)
bf = json.load(open(OUT / "botfree_stats.json", encoding="utf-8"))
net = json.load(open(OUT / "network_homophily.json", encoding="utf-8"))
core = json.load(open(OUT / "core_stats.json", encoding="utf-8"))
ml = json.load(open(OUT / "ml_stance.json", encoding="utf-8"))
nul = json.load(open(OUT / "null_model.json", encoding="utf-8"))

checks = [
    ("Reddit N", f"{bf['n_reddit']:,}".replace(",", "{,}")),
    ("YouTube N", f"{bf['n_youtube']:,}".replace(",", "{,}")),
    ("total N", f"{bf['n_total']:,}".replace(",", "{,}")),
    ("Reddit mean compound", f"{bf['sentiment']['reddit_mean']:.3f}".lstrip("-")),
    ("YouTube mean compound", f"{bf['sentiment']['youtube_mean']:.3f}"),
    ("Reddit Cramer V", f"{bf['sentiment']['by_stance']['reddit']['cramers_v']:.3f}"),
    ("YouTube Cramer V", f"{bf['sentiment']['by_stance']['youtube']['cramers_v']:.3f}"),
    ("Newman r", f"{abs(net['newman_assortativity']):.3f}"),
    ("cross-partisan %", f"{net['cross_partisan_reply_rate']*100:.1f}"),
    ("reply edges", f"{net['n_edges']:,}".replace(",", "{,}")),
    ("active users", f"{core['polarization']['n_active_users']:,}".replace(",", "{,}")),
    ("mean consistency", f"{core['polarization']['mean_consistency']:.3f}"),
    ("thread homophily", f"{core['polarization']['thread']['mean']:.3f}"),
    ("reply homophily", f"{core['polarization']['reply_network']['mean']:.3f}"),
    ("null z", str(abs(int(nul["z"])))),
    ("raw gap", f"{ml['transfer_gap']['raw']:.3f}"),
    ("matched gap", f"{ml['transfer_gap']['length_matched']:.3f}"),
]
for name, val in checks:
    print(f"  {'FOUND  ' if val in src else 'MISSING'}  {name:24s} -> {val}")
