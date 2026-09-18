# -*- coding: utf-8 -*-
"""Port the manuscript onto the Springer Nature sn-jnl class for submission to
the Journal of Computational Social Science.

Journal requirements applied (link.springer.com/journal/42001/submission-guidelines):
  * sn-jnl class, [iicol] formatting option
  * numbered [n] citations            -> sn-basic + Numbered
  * abstract 150-250 words            -> trimmed from 299
  * 4 to 6 keywords                   -> added
  * decimal headings, max three levels-> unstar \subsubsection*
  * single .tex document              -> \input{rq2_body} inlined
  * Statements and Declarations block -> added before references
"""
import re
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\nitro 5\Desktop\Projects\israel_hamas_discourse_analysis")
SRC = ROOT / "latex_script"
TPL = ROOT / "Springer_Nature_LaTeX_Template"
OUT = ROOT / "springer_submission"

GITHUB = "https://github.com/akshat3144/israel-hamas-discourse-analysis"

# --------------------------------------------------------------------------
# 1. preamble
# --------------------------------------------------------------------------
PREAMBLE = r"""%% Journal of Computational Social Science
%% Springer Nature LaTeX template, two-column ("iicol") formatting option,
%% Basic Springer Nature reference style with numbered citations.

\documentclass[pdflatex,sn-basic,Numbered,iicol]{sn-jnl}

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{booktabs}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{manyfoot}%
\usepackage{enumitem}%

\raggedbottom

\begin{document}

\title[Clustered by Thread, Arguing Across the Divide]{Clustered by Thread,
Arguing Across the Divide: Echo Chambers and the Choice of Measure in
Israel-Hamas Discourse on Reddit and YouTube}

%% TODO(authors): add each author's ORCID with \orcid{...} and confirm emails.
\author*[1]{\fnm{Akshat} \sur{Gupta}}\email{akshat.gupta@plaksha.edu.in}
\author[1]{\fnm{Raghav} \sur{Sarna}}\email{raghav.sarna@plaksha.edu.in}
\author[1]{\fnm{Arsh} \sur{Arora}}\email{arsh.arora@plaksha.edu.in}
\author[2]{\fnm{Uku} \sur{Kangur}}\email{uku.kangur@ut.ee}
\author[3]{\fnm{Rajesh} \sur{Sharma}}\email{rajesh.sharma@plaksha.edu.in}

\affil*[1]{\orgdiv{Department of Computer Science and Artificial Intelligence},
\orgname{Plaksha University}, \orgaddress{\city{Mohali}, \postcode{140306},
\state{Punjab}, \country{India}}}

\affil[2]{\orgdiv{Institute of Computer Science}, \orgname{University of Tartu},
\orgaddress{\city{Tartu}, \country{Estonia}}}

%% TODO(authors): affiliations 1 and 3 are both Plaksha University but carry
%% different unit names, as supplied. Merge them into one \affil if they are the
%% same unit.
\affil[3]{\orgdiv{School of AI and Computer Science}, \orgname{Plaksha University},
\orgaddress{\city{Mohali}, \postcode{140306}, \state{Punjab}, \country{India}}}

\abstract{ABSTRACT_HERE}

\keywords{Computational social science, Political polarization, Echo chambers,
Stance detection, Social media platforms, Annotation validation}

\maketitle

%% five authors across three affiliations leave too little room below the front
%% matter for a balanced two-column start, so the body opens on the next page.
%% \newpage would only move to the second column, hence \clearpage.
\clearpage
"""

ABSTRACT = r"""The Israel-Hamas conflict has unfolded not only through military
and diplomatic events but also through continuous public argument online. We
analyse 1{,}382{,}896 stance-labelled comments, 1{,}004{,}629 from Reddit and
378{,}267 from YouTube, over a matched 7~October~2023 to 7~May~2024 window, to ask
how platform architecture shapes political discourse. Three findings are not
predictable from platform design alone. First, emotional tone is politicised on
Reddit, where both partisan camps are negative and only neutral comments are
positive, but largely decoupled from stance on YouTube. Second, negativity and
toxicity are related but not interchangeable: 88.5\% of toxic Reddit comments are
negative, yet only 15.5\% of negative comments are toxic, and once sentiment is
controlled the apparent toxicity gap between the two partisan camps disappears.
Third, Reddit's reply network is disassortative by stance (Newman $r=-0.184$
against a degree-preserving null centred on zero), with 62.4\% of partisan replies
crossing the divide, yet the same users appear strongly clustered when measured by
thread co-membership. This locates the long-running echo-chamber disagreement in
the choice of measure rather than in the platform. Stance classifiers transfer
poorly across platforms, and this survives length matching, so it reflects
vocabulary rather than comment length. We release a reproducible LLM-assisted
labelling pipeline together with a three-annotator human validation
(Krippendorff's $\alpha=0.796$), which finds stance accuracy of $0.836$ on Reddit
but $0.618$ on YouTube, and we subject the platform contrast to three adversarial
checks built from that validation, all of which it survives."""

# --------------------------------------------------------------------------
# 2. back matter
# --------------------------------------------------------------------------
BACKMATTER = r"""
\backmatter

\bmhead{Acknowledgements}
%% TODO(authors): name the funding project and grant number. Springer requires
%% funding organisations to be written out in full.
This work was supported by [FUNDING BODY, PROJECT NAME AND GRANT NUMBER].

\section*{Statements and Declarations}

\bmhead{Funding}
%% TODO(authors): must match the Acknowledgements above and the funding entered
%% in the submission system, which feeds the CrossMark record.
This work was supported by [FUNDING BODY, PROJECT NAME AND GRANT NUMBER].

\bmhead{Competing interests}
The authors have no competing interests to declare that are relevant to the
content of this article.

\bmhead{Ethics approval and consent to participate}
%% TODO(authors): confirm this matches Plaksha University policy before submitting.
This study analyses publicly available user-generated comments retrieved through
the Arctic Shift Reddit archive and the YouTube Data API v3. No interaction with
human participants took place and no private, restricted or personally
identifying content was collected. Comment text is reported only in aggregate and
no individual account is named. Secondary analysis of public data of this kind
does not require review by an institutional ethics committee.

\bmhead{Consent for publication}
Not applicable.

\bmhead{Data availability}
The labelling pipeline, the analysis code, the revision scripts, the three
completed human validation annotation sets, the scoring scripts and the
video-level metadata (title, description and channel for all 2{,}637 videos) are
available at \url{GITHUB_HERE}. Reddit comments were obtained from the
Arctic Shift archive and YouTube comments through the YouTube Data API v3; both
are subject to the respective platform terms of service, so derived identifiers
rather than raw account handles are released.

\bmhead{Code availability}
All code is available at \url{GITHUB_HERE}.

\bmhead{Author contributions}
%% TODO(authors): correct this split before submitting. It is a placeholder.
A.G. designed the study, implemented the data collection and stance labelling
pipeline, and drafted the manuscript. R.Sa. implemented the topic and vocabulary
analysis. A.A. implemented the toxicity analysis. A.G., R.Sa. and A.A.
independently annotated the 270-comment human validation sample. U.K. and R.Sh.
supervised the work, reviewed the methodology and revised the manuscript
critically for important intellectual content. All authors read and approved the
final manuscript.

\bmhead{Use of AI-assisted technologies}
A large language model (Claude, Anthropic) was used for language editing and for
drafting assistance during preparation of this manuscript. All research design,
data collection, analysis, interpretation and validation were performed by the
authors, who reviewed and edited the full text and take responsibility for its
content. The distinct use of a large language model as a measurement instrument
in the stance annotation pipeline, whose accuracy is measured rather than
assumed, is described in Section~\ref{sec:label-validation}.

\bibliography{refs}

\end{document}
"""


# --------------------------------------------------------------------------
def build_body():
    txt = (SRC / "paper.tex").read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace("\r\n", "\n")

    start = txt.index(r"\section{Introduction}")
    end = txt.index(r"\bibliographystyle")
    body = txt[start:end]

    # Springer: one .tex file only, no \input
    def inline(m):
        f = SRC / (m.group(1) + ".tex")
        return f.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").rstrip()

    while r"\input{" in body:
        body = re.sub(r"\\input\{([^}]+)\}", inline, body)

    # the availability statement moves into the Declarations block
    body = re.sub(
        r"% =+\n\\section\*\{Data and Code Availability\}\n% =+\n.*?(?=\n\n|\Z)",
        "", body, flags=re.S)

    # decimal headings: unstar third-level heads so they are numbered
    body = body.replace(r"\subsubsection*{", r"\subsubsection{")

    # the class manages float placement; [H] needs the float package and is
    # discouraged by Springer production
    body = re.sub(r"\\begin\{(figure|table)\}\[H\]", r"\\begin{\1}[t]", body)

    # label the validation section so the AI declaration can cross-reference it
    body = body.replace(r"\subsection{Label Validation}",
                        "\\subsection{Label Validation}\\label{sec:label-validation}")

    # graphicspath is gone; figures sit beside the .tex
    body = body.replace("{paperfigs/", "{")

    # the Springer column is narrower than the old layout; four tables need to
    # be tightened to fit it
    narrow = [
        # tab:context
        (r"\begin{tabular}{@{}lcc@{}}" + "\n    \\toprule\n"
         r"    & Stance accuracy & Irrelevant given \\",
         r"\begin{tabular}{@{}lcc@{}}" + "\n    \\toprule\n"
         r"    & Stance acc. & Irrelevant given \\"),
        # tab:human
        (r"\begin{tabular}{lrr}", r"\begin{tabular}{@{}lrr@{}}"),
        (r"Inter-annotator $\kappa$ (mean pairwise) & 0.813  & 0.774  \\",
         r"Inter-annotator $\kappa$ (pairwise) & 0.813 & 0.774 \\"),
        (r"Model accuracy, stance classes           & 0.750  & 0.562  \\",
         r"Model accuracy, stance classes & 0.750 & 0.562 \\"),
        (r"Irrelevant content retained              & 18.9\% & 15.6\% \\",
         r"Irrelevant content retained & 18.9\% & 15.6\% \\"),
        (r"Stance accuracy on the remainder         & 0.836  & 0.618  \\",
         r"Stance accuracy, remainder & 0.836 & 0.618 \\"),
        # tab:tox
        (r"Attribute & Reddit & YouTube & rank-biserial $r$ \\",
         r"Attribute & Reddit & YouTube & rank-bis. $r$ \\"),
        # tab:ml
        (r"\begin{tabular}{llcc}", r"\begin{tabular}{@{}llcc@{}}"),
    ]
    for old, new in narrow:
        if old in body:
            body = body.replace(old, new, 1)
        else:
            print("  NOTE: narrow-table patch did not match:", old.splitlines()[0][:50])

    # the four tightened tables drop a size step
    for lbl in ("tab:context", "tab:human", "tab:tox", "tab:ml"):
        i = body.find("\\label{" + lbl + "}")
        if i < 0:
            print("  NOTE: label not found:", lbl)
            continue
        j = body.rfind(r"\begin{table}", 0, i)
        seg = body[j:i]
        body = body[:j] + seg.replace(r"\small", r"\footnotesize"
                                      "\n  \\setlength{\\tabcolsep}{3pt}", 1) + body[i:]

    # drop hand-tuned spacing that fought the old class
    body = re.sub(r"\n\s*\\vspace\{[^}]*\}\s*(?=\n)", "", body)
    body = re.sub(r"^% =+$\n", "", body, flags=re.M)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def main():
    OUT.mkdir(exist_ok=True)

    # class + bibliography style + bibliography
    for f in ("sn-jnl.cls", "sn-basic.bst"):
        shutil.copy2(TPL / f, OUT / f)
    shutil.copy2(SRC / "refs.bib", OUT / "refs.bib")

    body = build_body()

    figs = sorted(set(re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", body)))
    for fig in figs:
        name = fig if fig.lower().endswith(".png") else fig + ".png"
        src = SRC / "paperfigs" / name
        if src.exists():
            shutil.copy2(src, OUT / name)
        else:
            print("  MISSING FIGURE:", name)

    tex = (PREAMBLE.replace("ABSTRACT_HERE", " ".join(ABSTRACT.split()))
           + "\n\n" + body + "\n"
           + BACKMATTER.replace("GITHUB_HERE", GITHUB))

    (OUT / "sn_paper.tex").write_text(tex, encoding="utf-8", newline="\n")

    words = len(re.sub(r"\\[a-zA-Z]+|[{}$\\,]", " ", ABSTRACT).split())
    print(f"  abstract words : {words}  (JCSS allows 150-250)")
    print(f"  figures copied : {len(figs)}")
    print(f"  wrote          : {OUT / 'sn_paper.tex'}")


if __name__ == "__main__":
    main()
