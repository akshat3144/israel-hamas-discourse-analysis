# -*- coding: utf-8 -*-
"""Check every reference in the manuscript against Crossref.

Anti-hallucination pass: for each bib entry with a DOI, resolve it and compare
the registered title and first author against what we cite. Entries without a
DOI (books, arXiv preprints, reports) are listed for manual checking.
"""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\nitro 5\Desktop\Projects\israel_hamas_discourse_analysis")
BIB = ROOT / "springer_submission" / "refs.bib"
OUT = ROOT / "references" / "crossref_check.json"

UA = "reference-audit (mailto:akshat.gupta.ug23@plaksha.edu.in)"


def parse_bib(text):
    """Very small BibTeX reader: enough for this file."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|\s*$)", body, re.S):
            v = " ".join(fm.group(2).split())
            fields[fm.group(1).lower()] = v
        entries.append({"key": key, "type": kind, **fields})
    return entries


def clean(s):
    s = re.sub(r"[{}\\&]", "", s or "")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split())


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]


def main():
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    print(f"parsed {len(entries)} entries\n")
    results = []

    for e in entries:
        rec = {"key": e["key"], "doi": e.get("doi"), "cited_title": e.get("title")}
        if not e.get("doi"):
            rec["status"] = "NO_DOI_MANUAL"
            print(f"  [ ? ] {e['key']:18} no DOI, manual check")
            results.append(rec)
            continue

        try:
            m = crossref(e["doi"])
        except urllib.error.HTTPError as ex:
            rec["status"] = f"DOI_NOT_FOUND_{ex.code}"
            print(f"  [!!!] {e['key']:18} DOI DOES NOT RESOLVE ({ex.code})")
            results.append(rec)
            time.sleep(0.5)
            continue
        except Exception as ex:                      # network trouble, not a verdict
            rec["status"] = "ERROR_" + type(ex).__name__
            print(f"  [err] {e['key']:18} {ex}")
            results.append(rec)
            time.sleep(0.5)
            continue

        reg_title = (m.get("title") or [""])[0]
        reg_year = None
        for f in ("published-print", "published-online", "issued"):
            if m.get(f, {}).get("date-parts", [[None]])[0][0]:
                reg_year = m[f]["date-parts"][0][0]
                break
        auth = m.get("author") or []
        reg_first = auth[0].get("family", "") if auth else ""

        ct, rt = clean(e.get("title")), clean(reg_title)
        shared = len(set(ct.split()) & set(rt.split()))
        title_ok = shared >= max(3, int(0.6 * len(set(ct.split()))))

        cited_first = clean(e.get("author", "").split(" and ")[0].split(",")[0])
        author_ok = (not cited_first) or (clean(reg_first) in cited_first) \
            or (cited_first in clean(reg_first))

        rec.update({
            "status": "OK" if (title_ok and author_ok) else "MISMATCH",
            "registered_title": reg_title,
            "registered_year": reg_year,
            "registered_first_author": reg_first,
            "registered_container": (m.get("container-title") or [""])[0],
            "cited_year": e.get("year"),
            "title_match": title_ok,
            "author_match": author_ok,
            "url": m.get("URL"),
        })
        flag = "OK " if rec["status"] == "OK" else "!!!"
        yr = "" if str(reg_year) == str(e.get("year")) else f"  YEAR cited {e.get('year')} vs registered {reg_year}"
        print(f"  [{flag}] {e['key']:18} {reg_title[:52]}{yr}")
        results.append(rec)
        time.sleep(0.4)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import urllib.parse
    main()
