# -*- coding: utf-8 -*-
"""
Reposition the RQ3 contribution against De Francisci Morales et al. (2021),
who already established cross-cutting political interaction on Reddit, and add
the degree-preserving null-model result.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = HERE / "body_new.tex"
t = p.read_text(encoding="utf-8")


def sub(old, new, what):
    global t
    assert old in t, f"anchor missing: {what}"
    t = t.replace(old, new, 1)
    print(f"  updated: {what}")


# ---------------- abstract: soften the novelty claim
sub("""  Third, and contrary to the echo-chamber account, Reddit's \\emph{reply} network is
  \\emph{disassortative} by stance (Newman $r=-0.184$): 62.4\\% of partisan replies
  cross the divide, even though the same users appear strongly clustered when
  measured by thread co-membership.""",
    """  Third, Reddit's \\emph{reply} network is \\emph{disassortative} by stance
  (Newman $r=-0.184$ against a degree-preserving null of $r\\approx0$): 62.4\\% of
  partisan replies cross the divide. This replicates, in a geopolitical-conflict
  corpus, the cross-cutting interaction reported for US party politics; our
  addition is that the \\emph{same users} appear strongly clustered when measured by
  thread co-membership, locating the long-running echo-chamber disagreement in the
  choice of measure rather than in the platform.""",
    "abstract")

# ---------------- contributions
sub("""(3)~a network-level correction to the echo-chamber
account, showing that thread co-membership and actual reply behaviour give opposite
answers, with the reply graph disassortative by stance; and""",
    """(3)~evidence that thread co-membership and actual
reply behaviour give \\emph{opposite} answers on the same users, which reconciles the
conflicting echo-chamber and no-echo literatures as a measurement artifact rather
than a substantive disagreement; and""",
    "contributions")

# ---------------- related work: acknowledge the prior result up front
sub("""Cinelli
et al.~\\cite{cinelli2021} demonstrated that echo-chamber strength is itself
platform-dependent, motivating our comparative design.""",
    """Cinelli
et al.~\\cite{cinelli2021} demonstrated that echo-chamber strength is itself
platform-dependent, motivating our comparative design. Directly relevant to our
RQ3, De Francisci Morales et al.~\\cite{defrancisci2021} reconstructed the political
interaction network of Trump and Clinton supporters on Reddit and found a
\\emph{preference} for cross-cutting replies over within-group ones, asymmetrically
so, against a degree-preserving rewiring null. We therefore do not claim
cross-cutting interaction on Reddit as a new result; we ask whether it holds in a
geopolitical-conflict setting rather than domestic party politics, and why it
coexists with strong echo-chamber readings of the same platform.""",
    "related work")

# ---------------- results: add the null model + attribute the prior finding
sub("""Both partisan groups fall \\emph{below} their
random-mixing baseline (Table~\\ref{tab:mixing}), and the graph as a whole is
disassortative, with Newman $r=\\mathbf{-0.184}$.""",
    """Both partisan groups fall \\emph{below} their
random-mixing baseline (Table~\\ref{tab:mixing}), and the graph as a whole is
disassortative, with Newman $r=\\mathbf{-0.184}$.

A marginal baseline is not sufficient here, because activity and popularity are
unevenly distributed across stances: if Pro-Palestine users are both more numerous
and more replied-to, some apparent cross-cutting would be mechanical. We therefore
repeat the test against a \\emph{degree-preserving} null that permutes reply targets,
holding every user's out-degree and in-degree exactly fixed. Over 500 permutations
the null assortativity is centred on zero ($\\bar{r}=+0.00004$, $\\mathrm{sd}=0.0014$),
so the observed $-0.184$ lies $z=-128$ from the null and is not an artifact of the
degree sequence. The per-stance excesses survive the same test (Pro-Palestine
$z=-126$, Pro-Israel $z=-118$, Neutral $z=+51$).""",
    "results: null model")

sub("""Echo-chamber claims resting on co-presence alone
can therefore invert the conclusion drawn from interaction.""",
    """Echo-chamber claims resting on co-presence alone
can therefore invert the conclusion drawn from interaction. The cross-cutting
direction itself is consistent with De Francisci Morales
et al.~\\cite{defrancisci2021}, who report the same preference - and the same
asymmetry, with one camp more eager to reply across the divide - for US party
politics; finding it again in a conflict corpus suggests the pattern is not
specific to domestic partisanship.""",
    "results: attribution")

# ---------------- discussion
sub("""Since homophily is defined on networks~\\cite{mcpherson2001,newman2003}, the
reply-graph measure is the appropriate one, and it indicates that Reddit in this
corpus functions as a contested arena rather than a set of sealed chambers.""",
    """Since homophily is defined on networks~\\cite{mcpherson2001,newman2003}, the
reply-graph measure is the appropriate one, and it indicates that Reddit in this
corpus functions as a contested arena rather than a set of sealed chambers. That
conclusion is not new: De Francisci Morales et al.~\\cite{defrancisci2021} reached it
for US party politics with a comparable degree-preserving null. What our data add
is the contrast, on identical users, with a co-presence measure that points the
other way - which suggests the persistent disagreement between echo-chamber and
no-echo results is substantially a disagreement about what to measure.""",
    "discussion")

p.write_text(t, encoding="utf-8")
print("\nrepositioned RQ3 against prior work")
