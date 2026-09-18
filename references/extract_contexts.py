# -*- coding: utf-8 -*-
"""Pull the sentence around every \\cite in the manuscript.

This is what each reference is being used to support, and therefore what has to
be findable in the source document.
"""
import json
import re
from pathlib import Path

ROOT = Path(r"C:\Users\nitro 5\Desktop\Projects\israel_hamas_discourse_analysis")
TEX = ROOT / "springer_submission" / "sn_paper.tex"
OUT = ROOT / "references" / "citation_contexts.json"


def strip_tex(s):
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\\(cite|ref|label)\{[^}]*\}", " ", s)
    s = re.sub(r"\\(emph|textbf|textit|texttt|text)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = s.replace("~", " ")
    s = re.sub(r"[{}$]", "", s)
    return " ".join(s.split())


def main():
    tex = TEX.read_text(encoding="utf-8").replace("\r\n", "\n")
    # body only, so the bibliography does not count as a use
    body = tex[tex.index(r"\section{Introduction}"):]

    flat = " ".join(body.split("\n"))
    sentences = re.split(r"(?<=[.!?])\s+", flat)

    ctx = {}
    for sent in sentences:
        for m in re.finditer(r"\\cite\{([^}]*)\}", sent):
            for key in (k.strip() for k in m.group(1).split(",")):
                ctx.setdefault(key, []).append(strip_tex(sent))

    OUT.write_text(json.dumps(ctx, indent=2), encoding="utf-8")

    for key in sorted(ctx):
        print(f"\n### {key}  ({len(ctx[key])} use{'s' if len(ctx[key]) > 1 else ''})")
        for s in ctx[key]:
            print("   " + (s[:300] + ("..." if len(s) > 300 else "")))


if __name__ == "__main__":
    main()
