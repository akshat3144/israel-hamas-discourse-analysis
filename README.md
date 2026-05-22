# Israel-Hamas Discourse Analysis: A Comparative Study of Reddit and YouTube

A comprehensive computational social science study analyzing political discourse around the Israel-Hamas conflict across two major platforms. This project examines how platform architecture shapes public conversation, emotional expression, narratives, polarization, and harmful speech patterns.

## Project Overview

**Research Focus:** Understanding the digital discourse landscape of the Israel-Hamas conflict through comparative analysis of Reddit (debate-centric) and YouTube (media-centric) communities.

**Core Objective:** Identify platform-specific discourse patterns, emotional expressions, narrative framings, and toxicity levels to understand how different platform cultures influence political communication.

---

## Research Questions

This project systematically addresses four interconnected research questions:

| RQ      | Question                                          | Focus                                                            |
| ------- | ------------------------------------------------- | ---------------------------------------------------------------- |
| **RQ1** | How does emotional tone differ between platforms? | Sentiment polarity, subjectivity, platform culture effects       |
| **RQ2** | What distinct topics and narratives emerge?       | Discourse framing, narrative patterns, thematic differences      |
| **RQ3** | Do echo chambers exist and are users polarized?   | User behavior consistency, network analysis, stance polarization |
| **RQ4** | Which platform/stance harbors most toxic speech?  | Toxicity detection, harmful content patterns, demographic bias   |

---

## Module Breakdown & Outputs

| Folder                       | Purpose                             | Key Outputs                                                       | Insights                                                                                             |
| ---------------------------- | ----------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `01_data_preparation`        | Shared preparation step             | Cleaned datasets (CSV), EDA visualizations                        | Data quality baseline, engagement patterns                                                           |
| `02_emotional_tone_analysis` | RQ1: Emotional tone differences     | Sentiment distributions, statistical tests, comparative analysis  | Reddit is more negative; YouTube is more positive. Sentiment varies by platform and political stance |
| `03_topics_and_narratives`   | RQ2: Topic & narrative analysis     | Topic models (LDA/NMF), word clouds, thematic summaries           | Reddit emphasizes political/territorial framing; YouTube emphasizes emotional/religious framing      |
| `04_echo_chambers`           | RQ3: Polarization & network effects | Network diagrams, temporal analysis, ML predictions               | Echo chambers present; users maintain consistent stances; engagement varies by stance                |
| `05_toxicity_analysis`       | RQ4: Harmful speech patterns        | Toxicity scores by platform/stance, comparative toxicity analysis | Platform and stance differences in toxicity; identity attacks and threats vary                       |

## Repository Layout

Each analysis module follows a consistent, professional structure with scripts, notebooks, and dedicated outputs:

```
01_data_preparation/                    Data Cleaning & EDA
├── exploratory_analysis.ipynb           Interactive exploration
├── scripts/
│   └── eda_analysis.py                 Data quality & visualizations
├── outputs/
│   ├── eda_summary_report.txt
│   ├── reddit_processed.csv
│   └── youtube_processed.csv
└── main.py                              Entry point

02_emotional_tone_analysis/             Sentiment Analysis (RQ1)
├── sentiment_analysis.ipynb             Sentiment & statistics
├── scripts/
│   ├── sentiment_analysis.py           VADER & TextBlob scoring
│   └── statistical_tests.py            Statistical validation
├── outputs/
│   ├── sentiment_summary_report.txt
│   ├── reddit_with_sentiment.csv
│   ├── youtube_with_sentiment.csv
│   └── [8+ visualization plots]
└── main.py

03_topics_and_narratives/               Topic Modeling (RQ2)
├── topic_analysis.ipynb                Topic exploration
├── scripts/
│   ├── topic_modeling.py               LDA & NMF models
│   └── generate_wordclouds.py          Word clouds & frequency
├── outputs/
│   ├── topic_modeling_report.txt
│   ├── [7+ visualization plots]
│   └── wordclouds/
└── main.py

04_echo_chambers/                       Network & Polarization (RQ3)
├── network_analysis.ipynb              Network & ML analysis
├── scripts/
│   ├── advanced_analysis.py            Regression analysis
│   ├── structural_temporal_analysis.py Temporal patterns
│   ├── network_analysis.py             Echo chambers
│   └── ml_stance_classification.py     Stance prediction
├── outputs/
│   ├── regression_results.txt
│   ├── [15+ visualization plots]
│   └── ml_stance_classification/
└── main.py

05_toxicity_analysis/                   Toxicity Detection (RQ4)
├── toxicity_assessment.ipynb           Toxicity patterns
├── scripts/
│   └── toxicity_assessment.py         Perspective API analysis
├── outputs/
│   ├── perspective_analysis_report.txt
│   ├── reddit_perspective.csv
│   ├── youtube_perspective.csv
│   └── [4+ visualization plots]
└── main.py

data/                                   Raw Input Data
├── reddit_labeled.xlsx
└── youtube_labeled.xlsx

docs/                                   Reports & Analysis
├── Final_Comprehensive_Report.md
├── Phase3_Results_Summary.md
└── Phase4_Results_Summary.md

requirements.txt                        Dependencies
README.md                               Documentation
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

To fill each folder's `outputs/` directory, run the folder-level main scripts from the repository root:

```bash
python 01_data_preparation/main.py
python 02_emotional_tone_analysis/main.py
python 03_topics_and_narratives/main.py
python 04_echo_chambers/main.py
python 05_toxicity_analysis/main.py
```

Outputs are generated only inside each folder's `outputs/` directory. The `scripts/` files can still be run individually when needed.

## Key Methodologies

### Sentiment Analysis

- **VADER** (Valence Aware Dictionary and sEntiment Reasoner) for intensity-weighted sentiment
- **TextBlob** for supplementary polarity and subjectivity analysis
- Statistical tests: Chi-square, ANOVA, correlation analysis

### Topic Modeling

- **LDA (Latent Dirichletallocation)** with 5 topics per platform
- **NMF (Non-negative Matrix Factorization)** for alternative topic structures
- Word frequency analysis with domain-specific stopwords

### Network & Polarization Analysis

- User-stance consistency metrics
- Echo chamber detection via network clustering
- **ML Ensemble**: Voting classifier combining Logistic Regression, SVM, Random Forest, Gradient Boosting
- Temporal analysis of conversation dynamics

### Toxicity Detection

- **Google Perspective API** measuring:
  - Toxicity, Severe Toxicity, Identity Attack, Insult, Threat, Profanity
  - Stratified sampling to ensure stance representation

---

## Key Findings Summary

### RQ1: Emotional Tone (Sentiment)

- **Platform Contrast**: Reddit skews negative (45.1%); YouTube skews positive (44.8%)
- **Stance Effect**: Pro-Israel content is most negative on Reddit; Pro-Palestine content is most positive on YouTube
- **Implication**: Platform culture significantly influences how political stances are emotionally expressed

### RQ2: Topics & Narratives

- **Reddit Framing**: Political, territorial, analytical (state, land, country, rights)
- **YouTube Framing**: Emotional, religious, solidarity-based (free, support, Allah, God)
- **Stance-Specific**: Pro-Palestine emphasizes "genocide"; Pro-Israel emphasizes "civilians"
- **Implication**: Platforms support different discourse styles that appeal to different audiences

### RQ3: Echo Chambers & Polarization

- **Echo Chamber Presence**: Users cluster by stance with moderate to high consistency
- **Stance Stability**: User engagement and temporal patterns show platform-specific polarization
- **ML Predictions**: Cross-platform stance prediction shows learnable patterns despite platform differences
- **Implication**: Platforms amplify existing preferences while maintaining echo chamber structures

### RQ4: Toxicity & Harmful Speech

- **Platform Differences**: Toxicity distributions differ significantly between Reddit and YouTube
- **Stance-Toxicity Link**: Certain stances associated with higher toxicity; identity attacks correlate with stance
- **Implication**: Harmful content patterns are platform- and stance-dependent, enabling targeted moderation

---

## Notebooks

All notebooks provide interactive exploration and visualization of results. They are **display-only** (no file saving) and complement the scripts:

- `01_data_preparation/exploratory_analysis.ipynb` – Data overview and quality metrics
- `02_emotional_tone_analysis/sentiment_analysis.ipynb` – Sentiment exploration and statistics
- `03_topics_and_narratives/topic_analysis.ipynb` – Topic visualization and narrative analysis
- `04_echo_chambers/network_analysis.ipynb` – Network structures and ML predictions
- `05_toxicity_analysis/toxicity_assessment.ipynb` – Toxicity patterns and comparative analysis

Notebooks use markdown section headers for professional organization and include interpretive commentary.

---

## Project Architecture

This project follows a **modular, reproducible pipeline**:

1. **Independence**: Each module solves one research question independently
2. **Composability**: Later modules build on outputs from earlier modules
3. **Clarity**: Consistent naming, paths, and execution patterns across all modules
4. **Transparency**: Scripts perform file I/O; notebooks display results

---

## Notes

- **Generated Outputs**: Re-run the pipeline to regenerate all visualizations and reports in the `outputs/` directories
- **Data Location**: Place labeled Reddit and YouTube data in the `data/` folder (not tracked in version control)
- **Perspective API**: Requires valid API key for RQ4 toxicity analysis; set via `.env` file
- **Stance Labels**: P = Pro-Palestine, I = Pro-Israel, N = Neutral (consistent across all modules)
