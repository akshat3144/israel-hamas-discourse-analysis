# -*- coding: utf-8 -*-
"""
The human validation is done (3 annotators, 270 items, blind). Replace the
"protocol specified but not executed" section with the measured results, and
propagate the consequences to the abstract, contributions, discussion and
limitations.

Sources:
  06_revision/outputs/human_validation.json
  06_revision/outputs/validation_breakdown.json
  06_revision/outputs/label_noise_sensitivity.json

Run:  python latex_script/review3_validation.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body_new.tex"
NL = "\r\n"

START = "\\subsection{Label Validation}"
END = "\\subsection{Corpus Cleaning and Final Datasets}"

SECTION = NL.join([
    r"\subsection{Label Validation}",
    r"\label{sec:validation}",
    r"Label quality has two separable components. \emph{Reliability} asks whether the",
    r"annotator gives the same answer twice; \emph{accuracy} asks whether that answer is",
    r"right. We measure both: reliability on the labelled corpus itself, and accuracy",
    r"against three human annotators working blind.",
    r"",
    r"\subsubsection*{Test-retest reliability}",
    r"Comments were submitted in 50-item batches in file order, so two occurrences of an",
    r"identical comment text more than 50 rows apart were annotated in separate batches",
    r"and are independent re-annotations of the same item. Restricting further to repeats",
    r"within the \emph{same} post or video holds the surrounding context fixed. On the",
    r"High-confidence labels that enter the analytic corpus the annotator reproduces its",
    r"own label on \textbf{70.4\%} of 1{,}981 repeated Reddit items ($\kappa = 0.537$) and",
    r"\textbf{60.9\%} of 1{,}295 repeated YouTube items ($\kappa = 0.410$). Pooling all",
    r"confidence tiers lowers this to $\kappa = 0.522$ on Reddit and $0.254$ on YouTube,",
    r"so the stated confidence does track label stability.",
    r"",
    r"\subsubsection*{Human validation}",
    r"Three annotators independently labelled a stratified sample of \textbf{270}",
    r"comments, balanced across platform $\times$ stance $\times$ confidence tier (30 High",
    r"and 15 Medium/Low per stance per platform) drawn from the \emph{pre-filter} labelled",
    r"data. Annotators saw the comment and its context but never the model's label, and",
    r"worked from the same written guidelines given to the model. The gold standard is the",
    r"majority vote.",
    r"",
    r"The annotators agree substantially with each other: Krippendorff's",
    r"$\alpha = \textbf{0.796}$, Fleiss' $\kappa = 0.796$, pairwise Cohen's $\kappa$",
    r"between $0.761$ and $0.836$, and all three choose the same label on \textbf{78.1\%}",
    r"of items. Agreement is comparable on both platforms (mean pairwise $\kappa$ $0.813$",
    r"Reddit, $0.774$ YouTube). The task is therefore well posed and the gold standard is",
    r"not itself the weak link.",
    r"",
    r"Measured against that standard the model is weaker than its confidence tiers",
    r"suggest (Table~\ref{tab:human}). Two results matter and neither is comfortable.",
    r"",
    r"\textbf{The confidence filter does not screen out irrelevance.} Across the sample,",
    r"\textbf{17.0\%} of comments the model assigned a stance are judged \emph{irrelevant}",
    r"to the conflict by the human majority, and that rate is essentially identical in the",
    r"retained High tier (17.2\%) and the discarded Medium/Low tier (16.7\%). Roughly one",
    r"in six comments in the analytic corpus is therefore off-topic content carrying a",
    r"spurious stance label. The filter does work for its other purpose: among items whose",
    r"gold label is a stance, High-tier accuracy is $0.600$ against $0.433$ for",
    r"Medium/Low, a gap of $+16.7$ points that justifies keeping the filter.",
    r"",
    r"\textbf{Label quality is markedly worse on YouTube.} On High-tier items that humans",
    r"judge to carry a stance at all, the model matches the human gold standard on",
    r"\textbf{83.6\%} of Reddit comments but only \textbf{61.8\%} of YouTube ones. The",
    r"YouTube errors are concentrated in the Neutral class, which the model frequently",
    r"reads as Pro-Israel. Short, multilingual, formulaic comments are simply a harder",
    r"annotation problem, for the model as for anyone.",
    r"",
    r"\begin{table}[t]",
    r"  \centering",
    r"  \small",
    r"  \caption{Human validation against a three-annotator majority gold standard",
    r"  ($n=270$, blind). Stance accuracy and retained irrelevance are reported on the",
    r"  High-confidence tier, which is the corpus actually analysed.}",
    r"  \label{tab:human}",
    r"  \begin{tabular}{lrr}",
    r"    \toprule",
    r"     & Reddit & YouTube \\",
    r"    \midrule",
    r"    Inter-annotator $\kappa$ (mean pairwise) & 0.813  & 0.774  \\",
    r"    Model accuracy, stance classes           & 0.750  & 0.562  \\",
    r"    \midrule",
    r"    \multicolumn{3}{l}{\emph{High-confidence tier}} \\",
    r"    Irrelevant content retained              & 18.9\% & 15.6\% \\",
    r"    Stance accuracy on the remainder         & 0.836  & 0.618  \\",
    r"    \bottomrule",
    r"  \end{tabular}",
    r"  \vspace{2pt}",
    r"  \footnotesize Pooled across platforms: Krippendorff's $\alpha = 0.796$;",
    r"  78.1\% of items unanimous; stance-class macro-F1 $0.656$.",
    r"\end{table}",
    r"",
    r"\subsubsection*{Does the platform contrast survive the label-quality gap?}",
    r"Because label noise attenuates any stance-conditional association, and because our",
    r"central claim is that stance and sentiment are coupled on Reddit",
    r"(Cram\'er's $V = 0.149$) but not on YouTube ($V = 0.039$), the obvious objection is",
    r"that the contrast simply restates the difference in label quality. We test this",
    r"directly rather than argue about it. Taking the error structure measured on the",
    r"YouTube half of the validation sample, we corrupt 26\% of Reddit's stance labels",
    r"so that Reddit's accuracy falls to YouTube's $0.618$, and recompute the",
    r"association over 200 simulations.",
    r"",
    r"Reddit's $V$ falls from $0.149$ to $\textbf{0.082}$ $[0.081, 0.082]$, retaining 55\%",
    r"of the observed association and remaining \textbf{2.1 times} YouTube's $0.039$, far",
    r"outside the simulated interval. Differential label quality therefore accounts for",
    r"roughly \emph{half} of the raw gap between the platforms, and we accordingly present",
    r"the size of the contrast as an upper bound. Its existence and direction, however,",
    r"are not artifacts of annotation: Reddit retains a materially stronger",
    r"stance-sentiment coupling than YouTube even when its labels are degraded to",
    r"YouTube's measured quality.",
    r"",
    r"\subsubsection*{What this means for the results}",
    r"We report the validation in full rather than only where it flatters the pipeline.",
    r"Three consequences follow. First, the corpus should be read as approximately one in",
    r"six off-topic, which adds noise to every aggregate but does so at a similar rate on",
    r"both platforms (18.9\% versus 15.6\%) and so does not explain cross-platform",
    r"differences. Second, stance-conditional effects on YouTube are the least reliable",
    r"quantities in this paper and we treat them as directional only. Third, the",
    r"Reddit-based results, including the reply-network analysis that carries RQ3, rest on",
    r"labels that are $83.6\%$ accurate against a gold standard with $\alpha = 0.796$,",
    r"which we regard as adequate for the population-level comparisons made here. Sampling",
    r"code, blind sheets, the three completed annotation sets and the scoring script are",
    r"released with the corpus.",
    r"",
    "",
])


PROPAGATE = [
    # ---- abstract ----------------------------------------------------------
    (r"  pipeline, measure its test-retest reliability on repeated comments "
     r"($\kappa=0.54$ on Reddit, $0.41$ on YouTube) and show that its confidence filter "
     r"separates lexically coherent labels from discarded ones, and release a "
     r"human-validation protocol for the accuracy question that self-consistency "
     r"cannot settle.",
     r"  pipeline together with a three-annotator human validation "
     r"(Krippendorff's $\alpha=0.796$) which finds stance accuracy of $0.836$ on Reddit "
     r"but $0.618$ on YouTube; a simulation degrading Reddit's labels to YouTube's "
     r"measured quality shows the platform contrast is roughly half as large as it "
     r"first appears, but does not disappear."),

    # ---- contributions -----------------------------------------------------
    (r"pipeline with retained per-label justifications, a measured test-retest "
     r"reliability for that pipeline, and an accompanying human-validation",
     r"pipeline with retained per-label justifications, a completed three-annotator "
     r"human validation of it, and a quantified account of how far its error rate "
     r"bounds the conclusions, together with the released validation"),

    # ---- related work promise ---------------------------------------------
    (r"protocol (Section~\ref{sec:validation}) that reports the annotator's test-retest "
     r"reliability and the empirical value of the confidence filter now, and specifies "
     r"the human study needed to measure",
     r"protocol (Section~\ref{sec:validation}) reporting the annotator's test-retest "
     r"reliability, the empirical value of the confidence filter, and a three-annotator "
     r"human study measuring"),

    # ---- limitations -------------------------------------------------------
    (r"large language model. Its test-retest reliability is moderate "
     r"($\kappa = 0.54$ on Reddit, $0.41$ on YouTube; Table~\ref{tab:annrel}) and lower "
     r"on YouTube, so cross-platform comparisons of stance-conditional effects carry "
     r"unequal attenuation. We specify and release a human-validation protocol "
     r"(Section~\ref{sec:validation}), but the human annotation is not yet complete, so "
     r"label \emph{accuracy}, as distinct from reliability, is unmeasured and all",
     r"large language model whose accuracy we now measure rather than assume "
     r"(Section~\ref{sec:validation}): $0.836$ on Reddit but $0.618$ on YouTube against a "
     r"three-annotator gold standard, with roughly one comment in six carrying a stance "
     r"label that humans judge irrelevant. Cross-platform comparisons therefore carry "
     r"unequal attenuation; our simulation shows the headline contrast survives it at "
     r"about half its raw size, but YouTube stance-conditional effects should be read as "
     r"directional. Relabelling the corpus with a stronger annotator, and adding an "
     r"explicit relevance filter, are the clearest routes to tightening these"),
    (r"findings remain provisional in that respect. Second, the",
     r"estimates. Second, the"),
]


def main():
    with BODY.open("r", encoding="utf-8", newline="") as fh:
        raw = fh.read()

    if "tab:human" in raw:
        print("human validation already written in; nothing to do")
        return

    i, j = raw.index(START), raw.index(END)
    raw = raw[:i] + SECTION + raw[j:]
    print("  ok   replaced Label Validation section")

    missed = []
    for old, new in PROPAGATE:
        if old in raw:
            raw = raw.replace(old, new, 1)
            print(f"  ok   {old[:62]}")
        else:
            missed.append(old)
            print(f"  MISS {old[:62]}")

    with BODY.open("w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    print("\nall applied" if not missed else f"\n{len(missed)} pattern(s) missed")


if __name__ == "__main__":
    main()
