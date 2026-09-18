# -*- coding: utf-8 -*-
"""Where each reference can be obtained, and what to highlight inside it.

`pdf`    open-access copy, or None when no legal free full text exists
`terms`  phrases that should appear in the source if it really supports the
         claim the manuscript makes with it
"""

SOURCES = {
    # ---- open access ----------------------------------------------------
    "bail2018": dict(
        pdf="https://europepmc.org/articles/PMC6140520?pdf=render",
        terms=["increase political polarization", "opposing views", "liberals",
               "conservatives", "became substantially more conservative"]),
    "gilardi2023": dict(
        pdf="https://europepmc.org/articles/PMC10372638?pdf=render",
        terms=["outperforms crowd workers", "zero-shot", "accuracy",
               "MTurk", "annotation tasks"]),
    "cinelli2021": dict(
        pdf="https://europepmc.org/articles/PMC7936330?pdf=render",
        terms=["echo chamber", "Facebook", "Reddit", "Twitter", "Gab",
               "news consumption", "feed algorithm"]),
    "defrancisci2021": dict(
        pdf="https://www.nature.com/articles/s41598-021-81531-x.pdf",
        terms=["no echo", "cross-cutting", "rewiring", "null model",
               "Trump", "Clinton", "assortativity"]),
    "ngchow2025": dict(
        pdf="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0332746&type=printable",
        terms=["1.08 billion", "billion", "TikTok", "Reddit", "negative",
               "peaks"]),
    "blei2003": dict(
        pdf="https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf",
        terms=["generative probabilistic model", "Dirichlet", "mixture over",
               "latent topics"]),
    "pedregosa2011": dict(
        pdf="https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf",
        terms=["machine learning", "Python", "scikit-learn"]),
    "devlin2019": dict(
        pdf="https://aclanthology.org/N19-1423.pdf",
        terms=["bidirectional", "pre-training", "Transformer",
               "masked language model"]),
    "grootendorst2022": dict(
        pdf="https://arxiv.org/pdf/2203.05794",
        terms=["class-based TF-IDF", "clustering", "embeddings", "BERTopic"]),
    "antonakaki2025": dict(
        pdf="https://arxiv.org/pdf/2601.02367",
        terms=["Telegram", "Reddit", "BERTopic", "LDA", "emotion",
               "propaganda"]),
    "hutto2014": dict(
        pdf="https://ojs.aaai.org/index.php/ICWSM/article/download/14550/14399",
        terms=["VADER", "lexicon", "punctuation", "negation", "social media"]),
    "newman2003": dict(
        pdf="https://arxiv.org/pdf/cond-mat/0209450",
        terms=["assortative", "mixing", "discrete characteristics",
               "coefficient"]),
    "wulczyn2017": dict(
        pdf="https://arxiv.org/pdf/1610.08914",
        terms=["personal attack", "crowd", "annotat", "Wikipedia"]),
    "shugars2025": dict(
        pdf=None, why="Social Media + Society, gold open access but SAGE blocks automated download; open via doi.org/10.1177/20563051251332427",
        terms=["Reddit", "Twitter", "affordance", "visibility",
               "cross-cutting", "conversation"]),
    "boulianne2025": dict(
        pdf="https://europepmc.org/articles/PMC12599886?pdf=render",
        terms=["affordance", "anonymity", "character limit", "threading",
               "political"]),

    # ---- no legal free full text ---------------------------------------
    "mcpherson2001": dict(pdf=None, why="Annual Review of Sociology, subscription",
                          terms=["homophily", "birds of a feather", "similarity"]),
    "leeseung1999": dict(pdf=None, why="Nature, subscription",
                         terms=["non-negative matrix factorization", "parts-based"]),
    "flesch1948": dict(pdf=None, why="APA PsycNet, subscription",
                       terms=["reading ease", "readability"]),
    "ng2022": dict(pdf=None, why="Information Processing and Management, hybrid open access but Elsevier blocks automated download; open via doi.org/10.1016/j.ipm.2022.103070",
                   terms=["stance detection", "cross validation", "datasets",
                          "transfer"]),
    "santiago2025": dict(pdf=None, why="Springer book chapter, subscription",
                         terms=["Reddit", "emotion", "disgust", "sentiment"]),
    "sunstein2017": dict(pdf=None, why="Princeton University Press, book",
                         terms=["filtering", "self-selection", "echo chamber"]),
    "krippendorff2004": dict(pdf=None, why="Sage, book (2nd edition)",
                             terms=["alpha", "reliability", "agreement"]),

    # ---- web report -----------------------------------------------------
    "isd2023": dict(
        pdf=None, save_html=[
            "https://www.isdglobal.org/digital-dispatch/rise-in-antisemitism-on-both-mainstream-and-fringe-social-media-platforms-following-hamas-terrorist-attack/",
            "https://www.isdglobal.org/digital-dispatch/43-fold-increase-in-anti-muslim-youtube-comments-following-hamas-october-7-attack/",
        ],
        why="ISD Digital Dispatch, web publication (two dispatches)",
        terms=["50-fold", "fold increase", "antisemitic", "anti-Muslim",
               "YouTube"]),
}
