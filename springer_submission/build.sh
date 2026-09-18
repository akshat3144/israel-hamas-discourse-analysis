#!/bin/sh
# Rebuild the Springer submission and report compile quality.
#
# BibTeX and LaTeX output is captured, not discarded: an earlier version of this
# script sent bibtex to /dev/null and hid a real refs.bib syntax error that only
# surfaced on Overleaf.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")

cd "$ROOT"
python "$HERE/build_submission.py" 2>&1 | grep -E "abstract|figures|wrote|MISSING"

cd "$HERE"
pdflatex --enable-installer -interaction=nonstopmode sn_paper.tex > pass1.log 2>&1 || true
bibtex sn_paper > bibtex.log 2>&1 || true
pdflatex --enable-installer -interaction=nonstopmode sn_paper.tex > pass2.log 2>&1 || true
pdflatex --enable-installer -interaction=nonstopmode sn_paper.tex > pass3.log 2>&1 || true

echo "----------------------------------------"
grep -oE "Output written.*" sn_paper.log || echo "NO PDF PRODUCED"
echo "overfull   : $(grep -c Overfull sn_paper.log)"
echo "undef cite : $(grep -ci 'citation.*undefined' sn_paper.log)"
echo "undef ref  : $(grep -ci 'reference.*undefined' sn_paper.log)"

# bibtex failures are silent in the .log, so check its own output
BIBERR=$(grep -icE "error|I'm skipping|I was expecting" bibtex.log || true)
echo "bibtex errs: $BIBERR"
if [ "$BIBERR" != "0" ]; then
  echo "--- bibtex said ---"
  grep -iE -A2 "error|I'm skipping|I was expecting" bibtex.log | head -20
fi

WARN=$(grep -c "Difference (.*) between bookmark levels" sn_paper.log || true)
echo "bookmark warnings: $WARN  (cosmetic; sn-jnl places bmhead below section)"

rm -f pass1.log pass2.log pass3.log
