# Human validation package (reviewer point R7)

270 items sampled from the *pre-filter* labelled data so that we can validate two
things at once:

1. **Accuracy of the labels we keep** - the High-confidence P/I/N items
   (30 per stance per platform).
2. **Whether the confidence filter is justified** - the Medium/Low-confidence
   items we discarded (15 per stance per platform). If accuracy is materially
   higher in the High tier, the filter is empirically justified rather than
   assumed.

## Files

| File | What it is |
|---|---|
| `annotator_Raghav.xlsx`, `_Arsh`, `_Akshat` | one workbook per annotator - **this is what you fill in** |
| `validation_blind.csv` | the same 270 items, no model labels (source for the workbooks) |
| `validation_key.csv` | **do not open until everyone has finished** - holds the model's labels |

## Status: complete

All three annotators finished all 270 items. Results are in
`06_revision/outputs/human_validation.json`, `validation_breakdown.json` and
`label_noise_sensitivity.json`, and are reported in the paper's Label Validation
section. Headline: Krippendorff's alpha 0.796 between annotators; model stance
accuracy 0.836 on Reddit and 0.618 on YouTube; 17% of stance-labelled items judged
irrelevant by humans.

## Sentiment validation (same 270 items)

Two annotators (Akshat, Raghav) labelled the tone of each comment as positive,
negative or neutral, blind to every instrument's output; a third (Arsh) resolved
the items on which they disagreed, choosing between the two disputed labels.

| File | What it is |
|---|---|
| `sentiment_labels.csv` | round 2, the reported labels: both annotators, the adjudication, and the final gold label |
| `sentiment_labels_round1.csv` | round 1, superseded: annotators saw the comment but not the comment it replied to |

Round 1 was relabelled because the reading context was inadequate - most
Reddit items are replies, and the sheet did not show what they replied to. Round
2 adds the parent comment and, for YouTube, the video title
(`build_validation_context.py`, `make_sentiment_workbooks.py`). Both rounds lead to
the same conclusions; round 2 is reported. The workbooks themselves are not
released because that reading context includes content the platforms' terms do
not allow us to redistribute; item IDs join to `validation_blind.csv`.

Scoring: `python 06_revision/sentiment_validation.py [--round1]`, written to
`06_revision/outputs/sentiment_validation[_round1].json`. Headline (round 2):
Cohen's kappa 0.78 between annotators; accuracy against the gold label RoBERTa
0.68, VADER 0.52, TextBlob 0.32; human labels reproduce the stance-tone
association contrast (Cramer's V 0.355 Reddit vs 0.175 YouTube) but not a platform
difference in overall negativity.

## How to annotate (for re-running or extending the sample)

Open your workbook. It has two sheets:

- **Guidelines** - the label definitions, the rules, and one worked example.
- **Annotate** - 270 rows. Read the comment in column D, pick `P` / `I` / `N` / `R`
  from the dropdown in column E (yellow). Column F takes an optional note.

Rows you have not labelled stay highlighted, and the counter at the top right of
the sheet shows how many are done and how many are left. Save as `.xlsx` under the
same filename when finished.

Budget roughly **45-75 minutes**. The median comment is 84 characters; a handful
of Reddit ones run long.

Two rules that matter for the statistics:

- **Work independently.** Do not compare answers with the other annotators while
  labelling - independent agreement is the thing being measured.
- **Do not open `validation_key.csv`.** Seeing the model's label would bias you and
  invalidate the whole exercise.

At least **two** annotators must finish for scoring to run; three gives Fleiss'
kappa and a majority-vote gold standard, which is what the paper describes.

## Scoring

    python 06_revision/score_validation.py

Reads the workbooks directly (a filled-in `.csv` also works) and reports:

- inter-annotator agreement - pairwise Cohen's kappa, Fleiss' kappa,
  Krippendorff's alpha
- model-versus-human accuracy and macro-F1 against the majority-vote gold
  standard, overall and per platform
- **macro-F1 restricted to P/I/N**, which is the headline number. The sample was
  drawn only from items the model labelled P, I or N, so the model can never emit
  `R` here; including `R` in a 4-class macro-F1 would penalise it for a class it
  was never given the chance to predict.
- **irrelevant content retained** - the share of sampled items a human calls `R`
  although the model assigned a stance. This is the honest way to report the `R`
  disagreements, and doubles as a purity estimate for the analytic corpus.
- **High versus Medium/Low accuracy** - the comparison that converts the
  confidence filter from an assumption into a measured choice.

Results are written to `06_revision/outputs/human_validation.json`, which is what
the paper's Label Validation section will cite.

## Disclosure

Whoever annotates should be named in the paper, along with whether they were
involved in building the pipeline. Author-annotators are acceptable and common,
but only if disclosed - annotators are blind to the model's labels here, which is
the property that matters.
