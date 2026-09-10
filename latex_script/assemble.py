# -*- coding: utf-8 -*-
"""Assemble paper.tex = original preamble + revised body."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = r"\begin{document}"

src = (HERE / "paper.tex.bak").read_text(encoding="utf-8")
idx = src.find(MARKER)
if idx == -1:
    raise SystemExit("marker not found in paper.tex.bak")
preamble = src[:idx]

body = (HERE / "body_new.tex").read_text(encoding="utf-8")
if not body.lstrip().startswith(MARKER):
    raise SystemExit("body_new.tex must start with \\begin{document}")

out = preamble + body
assert out.count(MARKER) == 1, f"expected 1 begin{{document}}, got {out.count(MARKER)}"
assert out.count(r"\end{document}") == 1

(HERE / "paper.tex").write_text(out, encoding="utf-8")
print(f"assembled: preamble {preamble.count(chr(10))} lines + body "
      f"{body.count(chr(10))} lines = {out.count(chr(10))} lines")
