# -*- coding: utf-8 -*-
"""Package the manuscript sources into the two archives we need.

jcss_submission.zip  the editable source files Springer asks for at submission,
                     plus a PDF of the compiled output
overleaf_project.zip the same sources without the PDF, for uploading to Overleaf
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCES = ["sn_paper.tex", "refs.bib", "sn-jnl.cls", "sn-basic.bst"]

BUNDLES = {
    "jcss_submission.zip": SOURCES + ["sn_paper.pdf"],
    "overleaf_project.zip": SOURCES,
}


def build(name, sources):
    out = HERE / name
    files = [HERE / n for n in sources] + sorted(HERE.glob("f*.png"))

    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)

    print(f"  {name}: {len(files)} files, {out.stat().st_size / 1e6:.2f} MB")
    return files


def main():
    for name, sources in BUNDLES.items():
        files = build(name, sources)
    print("\n  contents (figures flattened beside the .tex):")
    for f in files:
        print(f"     {f.name}")


if __name__ == "__main__":
    main()
