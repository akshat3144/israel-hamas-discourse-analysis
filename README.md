# Israel-Hamas Discourse Analysis

A computational social science project comparing Reddit and YouTube discourse around the Israel-Hamas conflict.

The repository is organized directly by analysis stage and research question. Each folder has its own scripts, notebook, and clean `outputs/` directory.

## Research Questions

| Folder | Purpose | Main analyses |
| --- | --- | --- |
| `00_data_preparation_eda` | Shared preparation step | Cleaning, descriptive statistics, processed CSVs |
| `01_rq1_emotional_tone` | RQ1: How does emotional tone differ between Reddit and YouTube? | Sentiment analysis and statistical validation |
| `02_rq2_topics_narratives` | RQ2: What distinct topics and narratives emerge on each platform? | Topic modeling and word clouds |
| `03_rq3_echo_chambers` | RQ3: To what extent do echo chambers exist, and are users consistent in stance? | Network analysis, temporal/structural analysis, ML stance classification |
| `04_rq4_toxicity` | RQ4: Which platform and political stance harbor the most toxic speech? | Perspective API toxicity analysis |

## Repository Layout

```text
.
├── 00_data_preparation_eda/
│   ├── eda.ipynb
│   ├── scripts/
│   └── outputs/
├── 01_rq1_emotional_tone/
│   ├── sentiment_topic_modeling.ipynb
│   ├── scripts/
│   └── outputs/
├── 02_rq2_topics_narratives/
│   ├── topic_modeling.ipynb
│   ├── scripts/
│   └── outputs/
├── 03_rq3_echo_chambers/
│   ├── echo_chambers_and_ml.ipynb
│   ├── scripts/
│   └── outputs/
├── 04_rq4_toxicity/
│   ├── toxicity_analysis.ipynb
│   ├── scripts/
│   └── outputs/
├── data/
│   ├── reddit_labeled.xlsx
│   └── youtube_labeled.xlsx
├── docs/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

For RQ4, create a `.env` file with:

```text
PERSPECTIVE_API_KEY=your_key_here
```

## Run Order

To fill each folder's `outputs/` directory, run the folder-level runners from the repository root:

```bash
python 00_data_preparation_eda/run_all.py
python 01_rq1_emotional_tone/run_all.py
python 02_rq2_topics_narratives/run_all.py
python 03_rq3_echo_chambers/run_all.py
python 04_rq4_toxicity/run_all.py
```

Outputs are generated only inside each folder's `outputs/` directory. The `scripts/` files can still be run individually when needed.

## Notebooks

The notebooks sit directly inside each analysis folder:

- `00_data_preparation_eda/eda.ipynb`
- `01_rq1_emotional_tone/sentiment_topic_modeling.ipynb`
- `02_rq2_topics_narratives/topic_modeling.ipynb`
- `03_rq3_echo_chambers/echo_chambers_and_ml.ipynb`
- `04_rq4_toxicity/toxicity_analysis.ipynb`

The original complete EDA and sentiment/topic notebooks have been restored, with paths updated for this layout.
Notebooks are display-only: they show tables/plots inline and do not save files or populate output folders. Use `run_all.py` for reproducible saved artifacts.

## Notes

- Generated outputs were intentionally removed. Re-run the pipeline to regenerate them in the new locations.
- `data/` is expected to contain the labeled Reddit and YouTube Excel files, but it is ignored by git.
- Machine learning stance classification is under RQ3 because the report uses it in the polarization and echo-chamber section.
