# Human validation package (reviewer point R7)

270 items sampled from the *pre-filter* labelled data so that we can
validate two things at once:

1. **Accuracy of the labels we keep** - the High-confidence P/I/N items
   (30 per stance per platform).
2. **Whether the confidence filter is justified** - the Medium/Low-confidence
   items we discarded (15 per stance per platform). If accuracy is
   materially higher in the High tier, the filter is empirically justified
   rather than assumed.

## How to annotate
Each annotator opens their own `annotator_X.csv` and fills the `human_label`
column with exactly one of:

  P = Supports Palestine   I = Supports Israel
  N = Neutral / unclear    R = Irrelevant to the conflict

Use `00_data_collection_and_labeling/ANNOTATION_GUIDELINES.md` as the rulebook.
Annotate independently and do not discuss items while labelling - the whole point
is to measure independent agreement.

**Do not open `validation_key.csv`** until all annotators have finished; it
contains the model's labels.

## Scoring
Once at least two annotators are done:

    python 06_revision/score_validation.py

which reports inter-annotator agreement (Krippendorff's alpha / Cohen's kappa),
model-vs-human accuracy and macro-F1, a per-class confusion matrix, and the
High-vs-Medium/Low accuracy comparison.
