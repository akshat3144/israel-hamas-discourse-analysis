# -*- coding: utf-8 -*-
"""Where each cited reference was obtained.

One entry per reference in the manuscript's bibliography, and nothing else.
`pdf` is the open-access location it was fetched from; None means it was
downloaded by hand because the publisher blocks automated clients.

This file no longer carries search terms. Which passage to mark is decided in
mark_provenance.py, claim by claim, because keyword matching proved nothing.

Run audit_folder.py after changing the bibliography to catch drift between
this list, the compiled .bbl and the files in pdf/.
"""

SOURCES = {
    # ---- fetched automatically -------------------------------------------
    "antonakaki2025": dict(pdf="https://arxiv.org/pdf/2601.02367"),
    "bail2018": dict(pdf="https://europepmc.org/articles/PMC6140520?pdf=render"),
    "blei2003": dict(pdf="https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf"),
    "boulianne2025": dict(pdf="https://europepmc.org/articles/PMC12599886?pdf=render"),
    "cinelli2021": dict(pdf="https://europepmc.org/articles/PMC7936330?pdf=render"),
    "defrancisci2021": dict(pdf="https://www.nature.com/articles/s41598-021-81531-x.pdf"),
    "devlin2019": dict(pdf="https://aclanthology.org/N19-1423.pdf"),
    "gilardi2023": dict(pdf="https://europepmc.org/articles/PMC10372638?pdf=render"),
    "grootendorst2022": dict(pdf="https://arxiv.org/pdf/2203.05794"),
    "guerra2025": dict(pdf="https://arxiv.org/pdf/2412.10913"),
    "hayes2007": dict(
        pdf="https://www.asc.upenn.edu/sites/default/files/2021-03/"
            "Answering%20the%20Call%20for%20a%20Standard%20Reliability%20"
            "Measure%20for%20Coding%20Data.pdf"),
    "hofmann2026": dict(pdf="https://arxiv.org/pdf/2503.10648"),
    "hutto2014": dict(
        pdf="https://ojs.aaai.org/index.php/ICWSM/article/download/14550/14399"),
    "leeseung2000": dict(
        pdf="https://papers.nips.cc/paper_files/paper/2000/file/"
            "f9d1152547c0bde01830b7e8bd60024c-Paper.pdf"),
    "newman2003": dict(pdf="https://arxiv.org/pdf/cond-mat/0209450"),
    "ngchow2025": dict(
        pdf="https://journals.plos.org/plosone/article/file?"
            "id=10.1371/journal.pone.0332746&type=printable"),
    "pedregosa2011": dict(
        pdf="https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf"),
    "wulczyn2017": dict(pdf="https://arxiv.org/pdf/1610.08914"),

    # ---- open access, but the publisher refuses automated clients ---------
    "ng2022": dict(
        pdf=None,
        why="open access (CC BY-NC-ND); Elsevier returns 403 to scripts. "
            "Downloaded by hand from doi.org/10.1016/j.ipm.2022.103070"),
    "rosen2025": dict(
        pdf=None,
        why="gold open access; SAGE returns 403 to scripts. Downloaded by "
            "hand from doi.org/10.1177/20563051251383635"),
    "shugars2025": dict(
        pdf=None,
        why="gold open access (CC BY-NC); SAGE returns 403 to scripts. "
            "Downloaded by hand from doi.org/10.1177/20563051251332427"),
}
