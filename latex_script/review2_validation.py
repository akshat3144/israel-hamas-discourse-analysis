# -*- coding: utf-8 -*-
"""
Reviewer round 2: "In the Human Validation Protocol you mention the procedure, but
I don't seem to find any of these validation metrics brought out in the paper."

The human annotation is still outstanding, but two of the three things the protocol
promises can be measured on the labelled corpus itself, and now are:

  * test-retest reliability of the annotator on repeated comments
    (06_revision/annotation_reliability.py -> duplicate_consistency*)
  * whether the High-confidence filter is empirically justified
    (same script -> confidence_tier_validity)

This rewrites the section to report both, adds a results table, and keeps the
human protocol as the part that remains outstanding.

Run:  python latex_script/review2_validation.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body_new.tex"
NL = "\r\n"

START = "\\subsection{Human Validation Protocol}"
END = "\\subsection{Corpus Cleaning and Final Datasets}"

NEW = NL.join([
    r"\subsection{Label Validation}",
    r"\label{sec:validation}",
    r"Label quality has two separable components. \emph{Reliability} asks whether the",
    r"annotator gives the same answer twice; \emph{accuracy} asks whether that answer is",
    r"right. Reliability can be measured on the labelled corpus itself and we report it",
    r"here (Table~\ref{tab:annrel}); accuracy requires human judgement, and the protocol",
    r"for obtaining it is specified below but not yet executed.",
    r"",
    r"\subsubsection*{Test-retest reliability}",
    r"Comments were submitted in 50-item batches in file order, so two occurrences of an",
    r"identical comment text more than 50 rows apart were annotated in separate batches",
    r"and are independent re-annotations of the same item. Restricting further to repeats",
    r"within the \emph{same} post or video holds the surrounding context fixed, so the",
    r"model saw identical input both times. On the High-confidence labels that survive",
    r"into the analytic corpus, the annotator reproduces its own label on \textbf{70.4\%}",
    r"of 1{,}981 repeated Reddit items ($\kappa = 0.537$) and \textbf{60.9\%} of 1{,}295",
    r"repeated YouTube items ($\kappa = 0.410$). Pooling all confidence tiers lowers this",
    r"to $\kappa = 0.522$ on Reddit and $0.254$ on YouTube, so the stated confidence does",
    r"track label stability.",
    r"",
    r"These are moderate coefficients. They sit below Krippendorff's $0.667$ threshold",
    r"for tentative reliability~\cite{krippendorff2004}, which is why we do not treat the",
    r"labels as settled, and they are lower on YouTube, whose short multilingual comments",
    r"are the harder annotation problem. That asymmetry is a caveat rather than a",
    r"reassurance, and we state it plainly: label noise attenuates any stance-conditional",
    r"association, so lower YouTube reliability depresses YouTube's measured",
    r"stance-sentiment coupling in the same direction as our reported finding. The",
    r"reliability gap is nonetheless modest ($\kappa$ $0.54$ versus $0.41$) beside the",
    r"association gap it would have to explain (Cram\'er's $V$ of $0.149$ versus $0.039$,",
    r"and $0.264$ versus $0.063$ on the three-method consensus subset reported under",
    r"RQ1), so attenuation of that size is unlikely to account for the whole",
    r"contrast. It cannot be ruled out without human labels, which is what the",
    r"protocol below is for.",
    r"",
    r"\subsubsection*{Is the confidence filter doing any work?}",
    r"The corpus keeps only High-confidence labels, which discards 49.0\% of annotated",
    r"Reddit comments and 33.4\% of YouTube ones. To test whether that filter separates",
    r"anything real, we fit a TF-IDF logistic classifier on High-tier labels only and",
    r"compare how well it recovers held-out High-tier labels against the discarded",
    r"Medium/Low-tier ones. Recovery falls from macro-F1 $0.659$ to $0.514$ on Reddit and",
    r"from $0.474$ to $0.309$ on YouTube. Medium/Low items are thus substantially less",
    r"lexically coherent as a class, which makes the filter an empirical choice rather",
    r"than an assumption.",
    r"",
    r"\begin{table}[t]",
    r"  \centering",
    r"  \small",
    r"  \caption{Annotation reliability, measured on the labelled corpus. Test-retest",
    r"  compares independent re-annotations of the same comment text within the same",
    r"  thread or video; tier validity is the macro-F1 of a classifier trained on",
    r"  High-tier labels, evaluated on held-out High-tier items versus the discarded",
    r"  Medium/Low-tier ones.}",
    r"  \label{tab:annrel}",
    r"  \begin{tabular}{lrr}",
    r"    \toprule",
    r"     & Reddit & YouTube \\",
    r"    \midrule",
    r"    Labels at High confidence          & 51.0\%  & 66.6\%  \\",
    r"    Repeated items compared            & 1{,}981 & 1{,}295 \\",
    r"    Test-retest agreement (High tier)  & 70.4\%  & 60.9\%  \\",
    r"    Test-retest $\kappa$ (High tier)   & 0.537   & 0.410   \\",
    r"    Test-retest $\kappa$ (all tiers)   & 0.522   & 0.254   \\",
    r"    \midrule",
    r"    Tier validity, macro-F1 High       & 0.659   & 0.474   \\",
    r"    Tier validity, macro-F1 Medium/Low & 0.514   & 0.309   \\",
    r"    \bottomrule",
    r"  \end{tabular}",
    r"\end{table}",
    r"",
    r"\subsubsection*{What only human annotation can settle}",
    r"Consistency is not correctness: an annotator with a systematic bias can be perfectly",
    r"consistent. To measure accuracy we drew a stratified sample of \textbf{270} comments,",
    r"balanced across platform $\times$ stance $\times$ confidence tier (30 High and 15",
    r"Medium/Low per stance per platform) from the \emph{pre-filter} labelled data.",
    r"Annotators label each item blind against the published guidelines, and the scoring",
    r"script reports inter-annotator agreement (Krippendorff's~$\alpha$ and pairwise",
    r"$\kappa$), model-versus-human accuracy, macro-F1 and per-class confusion against a",
    r"majority-vote gold standard, and the High versus Medium/Low accuracy comparison.",
    r"Sampling code, blind sheets and the scoring script are released with the corpus. We",
    r"flag explicitly that this annotation is outstanding, so the accuracy of the labels,",
    r"as distinct from their reliability, remains unmeasured and all label-dependent",
    r"results are provisional in that respect.",
    r"",
    "",
])


def main():
    with BODY.open("r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    if "tab:annrel" in raw:
        print("validation results already present; nothing to do")
        return
    i, j = raw.index(START), raw.index(END)
    raw = raw[:i] + NEW + raw[j:]
    with BODY.open("w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    print("validation section rewritten with measured reliability")


if __name__ == "__main__":
    main()
