# -*- coding: utf-8 -*-
"""
Add the controlled video-context experiment to Label Validation.

Two arms over the same 131 YouTube validation items, identical model, prompt,
temperature and batch size; only the presence of the video title and
description differs. Source: 06_revision/context_experiment.py, results in
06_revision/outputs/context_experiment.json.

Run:  python latex_script/review6_context_test.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body_new.tex"
NL = "\r\n"

ANCHOR = ("multilingual and formulaic comments remain the harder annotation problem, "
          "but they" + NL + "are not the whole of it.")

SECTION = NL.join([
    ANCHOR,
    "",
    r"\subsubsection*{What video context is worth: a controlled test}",
    r"We test that attribution rather than assert it. Taking the 131 YouTube validation",
    r"items on which a human majority exists, we re-annotate them twice under identical",
    r"conditions, using the same model, provider, guidelines, temperature and batch size,",
    r"and varying only whether each item carries its video title and description. Both",
    r"arms use a different open-weight model from the deployed annotator, so what the",
    r"comparison isolates is the effect of context rather than of model choice.",
    r"",
    r"The no-context arm reproduces the corpus result exactly: $0.618$ against the human",
    r"gold standard on High-confidence stance items, the same figure the deployed",
    r"pipeline achieves. The shortfall is therefore a property of the prompt rather than",
    r"of the annotator model.",
    r"",
    r"Adding video context raises stance accuracy from $0.651$ to $\textbf{0.734}$ on",
    r"items humans judged to carry a stance, improving every class (Pro-Palestine $+12.5$",
    r"points, Pro-Israel $+8.0$, Neutral $+5.8$). It simultaneously degrades relevance",
    r"judgement: the share of human-irrelevant comments wrongly assigned a stance rises",
    r"from \textbf{18\%} to \textbf{73\%} (Table~\ref{tab:context}). Told that the video",
    r"concerns the conflict, the model infers that the comment must concern it too.",
    r"",
    r"\begin{table}[t]",
    r"  \centering",
    r"  \small",
    r"  \caption{What video context is worth. Both arms use the same model, prompt and",
    r"  settings on the same 131 items; only the video title and description differ.}",
    r"  \label{tab:context}",
    r"  \setlength{\tabcolsep}{4pt}",
    r"  \begin{tabular}{@{}lcc@{}}",
    r"    \toprule",
    r"    & Stance accuracy & Irrelevant given \\",
    r"    & ($n=109$) & a stance ($n=22$) \\",
    r"    \midrule",
    r"    No video context   & 0.651 & 18\% \\",
    r"    With video context & 0.734 & 73\% \\",
    r"    \bottomrule",
    r"  \end{tabular}",
    r"\end{table}",
    r"",
    r"This carries a design implication, and it is why we recommend separating the two",
    r"decisions rather than merely supplying more context. Relevance and stance want",
    r"video context in opposite directions: relevance is better judged on the comment",
    r"alone, stance with the video in view. A single four-way prompt cannot optimise",
    r"both, which also explains the retained irrelevance reported above. These figures",
    r"rest on 131 items, 22 of them irrelevant, and a single run; we report them as a",
    r"diagnosis of our own pipeline rather than as a general benchmark.",
])

PAIRS = [
    # the measured remedy in Limitations
    (r"The clearest routes to tightening these estimates are specific rather than "
     r"general: supplying video-level context to the YouTube annotator, which our "
     r"Reddit prompt had and our YouTube prompt lacked, and inserting an explicit "
     r"relevance decision ahead of",
     r"The clearest routes to tightening these estimates are specific rather than "
     r"general, and we measure both: supplying video-level context to the YouTube "
     r"annotator raises stance accuracy by roughly eight points but degrades relevance "
     r"judgement, so it must be paired with an explicit relevance decision ahead of"),

    # release statement
    (r"scoring scripts are released with the corpus.",
     r"scoring scripts are released with the corpus, as is the video-level metadata "
     r"(title, description and channel for all 2{,}637 videos) used in the context "
     r"experiment above."),
]


def main():
    with BODY.open("r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    if "tab:context" in raw:
        print("context experiment already present")
        return
    assert ANCHOR in raw, "anchor paragraph not found"
    raw = raw.replace(ANCHOR, SECTION, 1)
    print("  ok   inserted the controlled context test")
    missed = []
    for old, new in PAIRS:
        if old in raw:
            raw = raw.replace(old, new, 1)
            print(f"  ok   {old[:56]}")
        else:
            missed.append(old)
            print(f"  MISS {old[:56]}")
    with BODY.open("w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    print("\nall applied" if not missed else f"\n{len(missed)} missed")


if __name__ == "__main__":
    main()
