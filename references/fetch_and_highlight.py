# -*- coding: utf-8 -*-
"""Download each reference's open-access PDF and highlight the passages the
manuscript relies on.

Only publicly available copies are fetched: publisher open-access PDFs, PubMed
Central, arXiv, the ACL Anthology, JMLR and AAAI. Subscription-only items are
recorded as not retrievable rather than obtained by any other route.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import SOURCES  # noqa: E402

ROOT = Path(r"C:\Users\nitro 5\Desktop\Projects\israel_hamas_discourse_analysis")
REF = ROOT / "references"
PDF = REF / "pdf"
HTML = REF / "web"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0 Safari/537.36"),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
}


def download(url, dest):
    try:
        r = requests.get(url, headers=HEADERS, timeout=90, allow_redirects=True)
        r.raise_for_status()
        if r.content[:4] != b"%PDF":
            raise ValueError(f"not a PDF (got {r.headers.get('content-type','?')})")
        dest.write_bytes(r.content)
        return len(r.content)
    except Exception:
        # some open-access hosts reject this client but serve curl fine
        subprocess.run(
            ["curl", "-sL", "--max-time", "120", "-A",
             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
             "-o", str(dest), url],
            check=True, capture_output=True)
        if not dest.exists() or dest.read_bytes()[:4] != b"%PDF":
            dest.unlink(missing_ok=True)
            raise
        return dest.stat().st_size


def highlight(path, terms):
    """Yellow-highlight every term hit. Returns {term: [pages]}."""
    doc = fitz.open(path)
    hits = {}
    for term in terms:
        pages = []
        for pno in range(doc.page_count):
            page = doc[pno]
            rects = page.search_for(term, quads=False)
            if rects:
                for r in rects[:6]:          # cap per page, keeps files sane
                    a = page.add_highlight_annot(r)
                    a.set_colors(stroke=(1, 1, 0))
                    a.update()
                pages.append(pno + 1)
        if pages:
            hits[term] = pages
    if hits:
        doc.saveIncr()
    doc.close()
    return hits


def main():
    PDF.mkdir(parents=True, exist_ok=True)
    HTML.mkdir(parents=True, exist_ok=True)
    report = {}

    for key, spec in SOURCES.items():
        rec = {"key": key}

        if spec.get("save_html"):
            saved = []
            for i, url in enumerate(spec["save_html"], 1):
                try:
                    r = requests.get(url, headers=HEADERS, timeout=90)
                    r.raise_for_status()
                    p = HTML / f"{key}_{i}.html"
                    p.write_bytes(r.content)
                    saved.append(p.name)
                    print(f"  [html] {key:18} saved {p.name}")
                except Exception as ex:
                    print(f"  [FAIL] {key:18} {url[:60]} -> {ex}")
            rec.update(status="WEB_SAVED" if saved else "WEB_FAILED",
                       files=saved, why=spec.get("why"))
            report[key] = rec
            continue

        if not spec.get("pdf"):
            rec.update(status="NOT_RETRIEVABLE", why=spec.get("why"))
            print(f"  [ -- ] {key:18} {spec.get('why')}")
            report[key] = rec
            continue

        dest = PDF / f"{key}.pdf"
        if dest.exists() and dest.read_bytes()[:4] == b"%PDF":
            rec.update(status="OK", file=dest.name, bytes=dest.stat().st_size,
                       note="already downloaded and highlighted")
            print(f"  [keep] {key:18} {dest.stat().st_size/1024:7.0f} KB")
            report[key] = rec
            continue
        size = None
        err = None
        for url in (spec["pdf"], spec.get("alt")):
            if not url:
                continue
            try:
                size = download(url, dest)
                rec["source_url"] = url
                break
            except Exception as ex:
                err = f"{type(ex).__name__}: {ex}"
                time.sleep(1)

        if size is None:
            rec.update(status="DOWNLOAD_BLOCKED", error=err, why=spec.get("why"))
            print(f"  [FAIL] {key:18} {err}")
            report[key] = rec
            continue

        try:
            hits = highlight(dest, spec["terms"])
        except Exception as ex:
            hits = {}
            rec["highlight_error"] = str(ex)

        found = len(hits)
        total = len(spec["terms"])
        rec.update(status="OK", file=dest.name, bytes=size,
                   terms_found=found, terms_total=total, hits=hits)
        mark = "OK " if found else "!!!"
        print(f"  [{mark}] {key:18} {size/1024:7.0f} KB  {found}/{total} terms highlighted")
        report[key] = rec
        time.sleep(1)

    (REF / "fetch_report.json").write_text(json.dumps(report, indent=2),
                                           encoding="utf-8")
    print(f"\nwrote {REF / 'fetch_report.json'}")


if __name__ == "__main__":
    main()
