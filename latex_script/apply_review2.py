# -*- coding: utf-8 -*-
"""
Second-round reviewer edits (Uku Kangur, follow-up mail):
  1. Table 1 confidence intervals in +/- form rather than [lo, hi].
  2. Figure 11 caption spells out the R / Y axis abbreviations.
  3. Pro-Pal / Pro-Isr glossed where the stance scheme is defined.

All replacements are single-line so the file's CRLF endings are untouched.
Run:  python latex_script/apply_review2.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body_new.tex"

PAIRS = [
    # --- 1. +/- confidence intervals -------------------------------------
    (r"455{,}709 \quad 45.36\% [45.26, 45.46]",
     r"455{,}709 \quad 45.36\% $\pm$0.10\%"),
    (r"212{,}217 \quad 56.10\% [55.94, 56.26]",
     r"212{,}217 \quad 56.10\% $\pm$0.16\%"),
    (r"362{,}340 \quad 36.07\% [35.97, 36.16]",
     r"362{,}340 \quad 36.07\% $\pm$0.10\%"),
    (r"131{,}357 \quad 34.73\% [34.57, 34.88]",
     r"131{,}357 \quad 34.73\% $\pm$0.16\%"),
    (r"186{,}580 \quad 18.57\% [18.50, 18.65]",
     r"186{,}580 \quad 18.57\% $\pm$0.08\%"),
    (r"\phantom{0}9.17\% [\phantom{0}9.08, \phantom{0}9.27]",
     r"\phantom{0}9.17\% $\pm$0.10\%"),
    (r"  automated accounts removed). Stance shares carry bootstrapped 95\% CIs.}",
     r"  automated accounts removed). Stance shares are given as the point estimate "
     r"$\pm$ the half-width of a bootstrapped 95\% confidence interval "
     r"(4{,}000 resamples).}"),

    # --- 2. Figure 11 caption: spell out the R / Y axis labels -----------
    (r"  \caption{Cross-platform transfer remains far below within-platform performance",
     r"  \caption{Cross-platform transfer remains far below within-platform performance "
     r"after length matching. Axis labels give train$\rightarrow$test, with "
     r"\textbf{R} = Reddit and \textbf{Y} = YouTube: R$\rightarrow$R and "
     r"Y$\rightarrow$Y are within-platform, R$\rightarrow$Y and Y$\rightarrow$R "
     r"cross-platform.}%"),
    (r"  after length matching.}", r"  "),

    # --- 3. gloss the abbreviations used in the figures ------------------
    (r"scheme: Pro-Palestine~(P), Pro-Israel~(I), Neutral~(N) and Irrelevant~(R). Labelling",
     r"scheme: Pro-Palestine~(P), Pro-Israel~(I), Neutral~(N) and Irrelevant~(R). "
     r"Figures abbreviate the two partisan classes as \emph{Pro-Pal} (Pro-Palestine) "
     r"and \emph{Pro-Isr} (Pro-Israel). Labelling"),
]


def main():
    with BODY.open("r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    missed = []
    for old, new in PAIRS:
        if old in raw:
            raw = raw.replace(old, new)
            print(f"  ok   {old[:66]}")
        else:
            missed.append(old)
            print(f"  MISS {old[:66]}")
    with BODY.open("w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    print("\nall applied" if not missed else f"\n{len(missed)} pattern(s) missed")


if __name__ == "__main__":
    main()
