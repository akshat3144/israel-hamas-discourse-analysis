# -*- coding: utf-8 -*-
"""Package the editable source files Springer requires at submission.

The journal asks for the original source (all style files and figures) plus a
PDF of the compiled output, so both go in the archive.
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "jcss_submission.zip"

SOURCES = ["sn_paper.tex", "refs.bib", "sn-jnl.cls", "sn-basic.bst", "sn_paper.pdf"]


def main():
    figs = sorted(HERE.glob("f*.png"))
    files = [HERE / n for n in SOURCES] + figs

    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)

    print(f"  {OUT.name}: {len(files)} files, {OUT.stat().st_size / 1e6:.2f} MB")
    for f in files:
        print(f"     {f.name}")


if __name__ == "__main__":
    main()
