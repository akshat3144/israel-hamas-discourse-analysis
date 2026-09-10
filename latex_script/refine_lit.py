# -*- coding: utf-8 -*-
"""Sharpen positioning against the closest prior work; soften over-claims."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sub(path, old, new, what):
    p = HERE / path
    t = p.read_text(encoding="utf-8")
    assert old in t, f"ANCHOR MISSING ({path}): {what}"
    p.write_text(t.replace(old, new, 1), encoding="utf-8")
    print(f"  ok: {what}")


# --- 1. conflict-specific related work: precise differentiation
sub("rw_conflict.tex",
    """These
studies establish that the conflict produced intense and polarized digital
discourse, but neither compares Reddit and YouTube directly nor examines how
platform architecture shapes the form that discourse takes - the gap this work
addresses.""",
    """Notably, Antonakaki et al.\\ already report that Reddit
hosts the more reflective and deliberative discussion of the platforms they
compare, which is one reason we treat that contrast as an established premise
rather than a finding of ours.

These studies establish that the conflict produced intense and polarized digital
discourse, but they leave three gaps this work addresses. First, neither includes
\\textbf{YouTube}, so the media-centric reaction layer is absent from existing
cross-platform comparisons of this conflict. Second, neither assigns
\\textbf{per-comment stance} at scale, which is what allows sentiment, toxicity and
interaction to be analysed \\emph{conditional on} political position. Third, neither
examines \\textbf{interaction structure} - reply networks, homophily, or
cross-platform stance transfer - so the question of whether these communities
actually talk to each other is untouched.""",
    "rw_conflict differentiation")

# --- 2. soften the toxicity/sentiment novelty (the distinction is recognised)
sub("body_new.tex",
    """(2)~a demonstration that toxicity is a distinct and much rarer phenomenon
than negativity, and that the apparent partisan toxicity asymmetry is absorbed by
sentiment;""",
    """(2)~a quantification of how far toxicity and negativity
diverge in this corpus - the distinction is recognised in principle, but we measure
the overlap directly and show that the apparent partisan toxicity asymmetry is
absorbed once sentiment is controlled;""",
    "contributions: toxicity claim")

sub("body_new.tex",
    """They are related but far from interchangeable (Figure~\\ref{fig:jointtox}). Sentiment
and toxicity correlate moderately (Spearman $\\rho=-0.435$ on Reddit, $-0.240$ on
YouTube).""",
    """That toxicity and negative sentiment are distinct constructs is well recognised -
Perspective targets rudeness and disrespect rather than valence - but the two are
often used interchangeably in practice, and the size of the gap is rarely
quantified. In this corpus they are related but far from interchangeable
(Figure~\\ref{fig:jointtox}). Sentiment and toxicity correlate moderately
(Spearman $\\rho=-0.435$ on Reddit, $-0.240$ on YouTube).""",
    "results: toxicity framing")

print("\nliterature refinements applied")
