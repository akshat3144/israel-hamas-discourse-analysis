# -*- coding: utf-8 -*-
"""
Round-2 follow-ups to the new Label Validation section: surface the measured
reliability in the abstract, the contributions list, the related-work paragraph
that promises the protocol, and the limitations paragraph that disclaims it.

Run:  python latex_script/review2_xrefs.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body_new.tex"

PAIRS = [
    # abstract
    (r"  rather than comment length. We release a reproducible LLM-assisted labelling",
     r"  rather than comment length. We release a reproducible LLM-assisted labelling"),
    (r"  pipeline together with a human-validation protocol.",
     r"  pipeline, measure its test-retest reliability on repeated comments "
     r"($\kappa=0.54$ on Reddit, $0.41$ on YouTube) and show that its confidence filter "
     r"separates lexically coherent labels from discarded ones, and release a "
     r"human-validation protocol for the accuracy question that self-consistency "
     r"cannot settle."),

    # contributions (4)
    (r"pipeline with retained per-label justifications and an accompanying human-validation",
     r"pipeline with retained per-label justifications, a measured test-retest "
     r"reliability for that pipeline, and an accompanying human-validation"),

    # related work promise
    (r"human-validation protocol (Section~\ref{sec:validation}) measuring both",
     r"protocol (Section~\ref{sec:validation}) that reports the annotator's test-retest "
     r"reliability and the empirical value of the confidence filter now, and specifies "
     r"the human study needed to measure"),
    (r"model-human agreement and whether the confidence filter is justified.",
     r"model-human agreement."),

    # limitations
    (r"Several limitations qualify these results. First, stance labels were assigned by a",
     r"Several limitations qualify these results. First, stance labels were assigned by a"),
    (r"large language model; we specify and release a human-validation protocol",
     r"large language model. Its test-retest reliability is moderate "
     r"($\kappa = 0.54$ on Reddit, $0.41$ on YouTube; Table~\ref{tab:annrel}) and lower "
     r"on YouTube, so cross-platform comparisons of stance-conditional effects carry "
     r"unequal attenuation. We specify and release a human-validation protocol"),
    (r"(Section~\ref{sec:validation}), but the annotation is not yet complete, so all",
     r"(Section~\ref{sec:validation}), but the human annotation is not yet complete, so "
     r"label \emph{accuracy}, as distinct from reliability, is unmeasured and all"),
]


def main():
    with BODY.open("r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    missed = []
    for old, new in PAIRS:
        if old == new:
            continue
        if old in raw:
            raw = raw.replace(old, new, 1)
            print(f"  ok   {old[:64]}")
        else:
            missed.append(old)
            print(f"  MISS {old[:64]}")
    with BODY.open("w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    print("\nall applied" if not missed else f"\n{len(missed)} missed")


if __name__ == "__main__":
    main()
