# -*- coding: utf-8 -*-
"""Reposition RQ3 against De Francisci Morales et al. (2021) + add null model."""
from pathlib import Path

p = Path(__file__).resolve().parent / "body_new.tex"
t = p.read_text(encoding="utf-8")


def sub(old, new, what):
    global t
    assert old in t, f"ANCHOR MISSING: {what}"
    t = t.replace(old, new, 1)
    print(f"  ok: {what}")


# 1. contributions
sub("""(3)~a network-level correction to the echo-chamber account, showing that
thread co-membership and actual reply behaviour give opposite answers, with the
reply graph disassortative by stance; and""",
    """(3)~evidence that thread co-membership and actual reply
behaviour give \\emph{opposite} answers on the same users, which reframes the
long-running echo-chamber disagreement as substantially a question of measurement; and""",
    "contributions")

# 2. related work
sub("""Cinelli
et al.~\\cite{cinelli2021} demonstrated that echo-chamber strength is itself
platform-dependent, motivating our comparative design.""",
    """Cinelli
et al.~\\cite{cinelli2021} demonstrated that echo-chamber strength is itself
platform-dependent, motivating our comparative design. Most directly relevant to
our RQ3, De Francisci Morales et al.~\\cite{defrancisci2021} reconstructed the
political interaction network of Trump and Clinton supporters on Reddit and found a
\\emph{preference} for cross-cutting replies over within-group ones - asymmetrically
so - against a degree-preserving rewiring null. We therefore do not claim
cross-cutting interaction on Reddit as a new result. We ask instead whether it holds
in a geopolitical-conflict setting rather than domestic party politics, and why it
coexists with strong echo-chamber readings of the same platform.""",
    "related work")

# 3. results: degree-preserving null
sub("""Both partisan groups fall \\emph{below} their
random-mixing baseline (Table~\\ref{tab:mixing}), and the graph as a whole is
disassortative, with Newman $r=\\mathbf{-0.184}$.""",
    """Both partisan groups fall \\emph{below} their
random-mixing baseline (Table~\\ref{tab:mixing}), and the graph as a whole is
disassortative, with Newman $r=\\mathbf{-0.184}$.

A marginal baseline alone would not settle this, because activity and popularity are
unevenly distributed across stances: if Pro-Palestine users are both more numerous
and more replied-to, some apparent cross-cutting would be mechanical. We therefore
repeat the test against a \\emph{degree-preserving} null that permutes reply targets,
holding every user's out-degree and in-degree exactly fixed. Across 500 permutations
the null assortativity is centred on zero ($\\bar{r}=+0.00004$, $\\mathrm{sd}=0.0014$),
placing the observed $-0.184$ some $z=-128$ from the null; the per-stance excesses
survive the same test (Pro-Palestine $z=-126$, Pro-Israel $z=-118$, Neutral
$z=+51$). The cross-cutting pattern is therefore not an artifact of the degree
sequence.""",
    "results: null model")

# 4. results: attribution
sub("""resting on co-presence alone can therefore invert the conclusion drawn from
interaction.""",
    """resting on co-presence alone can therefore invert the conclusion drawn from
interaction. The cross-cutting direction itself agrees with De Francisci Morales
et al.~\\cite{defrancisci2021}, who report the same preference - and a similar
asymmetry, with one camp more eager to reply across the divide - for US party
politics; recovering it in a conflict corpus indicates the pattern is not confined
to domestic partisanship.""",
    "results: attribution")

# 5. discussion
sub("""Since homophily is defined on networks~\\cite{mcpherson2001,newman2003}, the
reply-graph measure is the appropriate one, and it indicates that Reddit in this
corpus functions as a contested arena rather than a set of sealed chambers.""",
    """Since homophily is defined on networks~\\cite{mcpherson2001,newman2003}, the
reply-graph measure is the appropriate one, and it indicates that Reddit in this
corpus functions as a contested arena rather than a set of sealed chambers. That
conclusion is not itself new - De Francisci Morales et al.~\\cite{defrancisci2021}
reached it for US party politics using a comparable degree-preserving null. What our
data add is the direct contrast, on identical users, with a co-presence measure that
points the other way, which suggests the persistent disagreement between
echo-chamber and no-echo findings is substantially a disagreement about what is
being measured.""",
    "discussion")

p.write_text(t, encoding="utf-8")
print("\nrepositioning complete")
