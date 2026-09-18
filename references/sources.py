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
        pdf=None, why="downloaded by hand: gold open access (CC BY-NC), SAGE blocks scripts",
        terms=["Reddit", "Twitter", "affordance", "visibility",
               "cross-cutting", "conversation"]),
    "boulianne2025": dict(
        pdf="https://europepmc.org/articles/PMC12599886?pdf=render",
        terms=["affordance", "anonymity", "character limit", "threading",
               "political"]),

    # ---- no legal free full text ---------------------------------------
    "mcpherson2001": dict(pdf=None, why="Annual Review of Sociology, subscription",
                          terms=["homophily", "birds of a feather", "similarity"]),
    "leeseung2000": dict(pdf=None, why="downloaded from NeurIPS proceedings (open)",
                         terms=["non-negative matrix factorization", "multiplicative",
                                "update rule"]),
    "flesch1948": dict(pdf=None, why="APA PsycNet, subscription",
                       terms=["reading ease", "readability"]),
    "ng2022": dict(pdf=None, why="downloaded by hand: open access (CC BY-NC-ND), Elsevier blocks scripts",
                   terms=["stance detection", "cross validation", "datasets",
                          "transfer"]),
    "santiago2025": dict(pdf=None, why="Springer book chapter, subscription",
                         terms=["Reddit", "emotion", "disgust", "sentiment"]),
    "hayes2007": dict(pdf=None, why="downloaded from UPenn Annenberg (open)",
                      terms=["alpha", "reliability", "agreement", "coding data"]),

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
