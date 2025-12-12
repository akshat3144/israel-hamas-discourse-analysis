# Israel-Hamas Discourse Analysis

A comprehensive computational social science study analyzing **19,362 comments** from Reddit and YouTube to understand how platform architecture shapes political discourse on the Israel-Hamas conflict.

## 📖 Research Overview

This project examines how social media platforms influence political discourse through:

- **Sentiment Analysis** - Emotional tone differences between platforms
- **Topic Modeling** - Distinct narratives and framing on each platform
- **Echo Chamber Detection** - Network analysis of user interactions
- **Toxicity Analysis** - Harmful speech patterns using Google's Perspective API
- **Machine Learning** - Stance classification and cross-platform prediction

### Research Questions

| RQ      | Question                                                                                    |
| :------ | :------------------------------------------------------------------------------------------ |
| **RQ1** | How does the emotional tone differ between debate-centric Reddit and media-centric YouTube? |
| **RQ2** | What distinct topics and narratives emerge on each platform?                                |
| **RQ3** | To what extent do echo chambers exist, and are users consistent in their stance?            |
| **RQ4** | Which platform—and which political stance—harbors the most toxic speech?                    |

## 🔑 Key Findings

- **Reddit** skews negative (45%) with complex, argumentative discourse
- **YouTube** skews positive (45%) with emotional, supportive comments
- **Pro-Israel** comments exhibit the highest toxicity levels on both platforms
- **Pro-Palestine** users show stronger echo chamber behavior
- Cross-platform ML models struggle due to distinct linguistic norms

## 📁 Project Structure

```
├── data/                          # Raw and labeled datasets
│   ├── reddit.xlsx
│   ├── reddit_labeled.xlsx
│   ├── youtube.xlsx
│   └── youtube_labeled.xlsx
├── eda_output/                    # Exploratory data analysis results
├── sentiment_output/              # Sentiment analysis results
├── topic_modeling_output/         # LDA/NMF topic modeling results
├── ml_output/                     # Machine learning model outputs
├── network_output/                # Echo chamber network visualizations
├── perspective_output/            # Toxicity analysis results
├── advanced_analysis_output/      # Statistical regression results
├── statistical_tests_output/      # Hypothesis testing results
├── word_cloud/                    # Word cloud visualizations
├── docs/                          # Documentation and reports
├── report_conversion/             # Report conversion utilities
│
├── eda_analysis.py                # Exploratory data analysis
├── sentiment_analysis.py          # VADER & TextBlob sentiment analysis
├── topic_modeling.py              # LDA & NMF topic modeling
├── ml_stance_classification.py    # Ensemble ML stance classifier
├── network_echo_chambers.py       # User interaction network analysis
├── perspective_analysis.py        # Google Perspective API toxicity
├── advanced_analysis.py           # Regression & complexity analysis
├── statistical_tests.py           # Statistical hypothesis testing
├── structural_temporal_analysis.py # Structural & temporal analysis
├── label_reddit_data.py           # Data labeling utilities
├── label_youtube_data.py          # Data labeling utilities
├── generate_wordclouds.py         # Word cloud generation
│
├── eda.ipynb                      # Interactive EDA notebook
├── sentiment_topic_modeling.ipynb # Sentiment & topic notebook
│
└── requirements.txt               # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/akshat3144/israel-hamas-discourse-analysis.git
   cd israel-hamas-discourse-analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (for Perspective API)

   ```bash
   # Create a .env file with your API keys
   GOOGLE_API_KEY=your_google_api_key
   ```

### Running the Analysis Pipeline

Execute the scripts in order:

```bash
# Phase 1: Exploratory Data Analysis
python eda_analysis.py

# Phase 2: Sentiment Analysis
python sentiment_analysis.py

# Phase 3: Topic Modeling
python topic_modeling.py

# Phase 4: Advanced Statistical Analysis
python advanced_analysis.py
python statistical_tests.py

# Phase 5: Machine Learning Classification
python ml_stance_classification.py

# Phase 6: Network & Echo Chamber Analysis
python network_echo_chambers.py

# Phase 7: Toxicity Analysis (requires API key)
python perspective_analysis.py

# Generate Word Clouds
python generate_wordclouds.py
```

## 📊 Methods & Tools

### Data Collection

- **Reddit**: 9,973 comments from relevant subreddits
- **YouTube**: 9,389 comments from news/political videos

### Analysis Techniques

| Phase     | Technique                             | Tools                       |
| :-------- | :------------------------------------ | :-------------------------- |
| EDA       | Descriptive statistics, distributions | Pandas, Matplotlib, Seaborn |
| Sentiment | VADER, TextBlob polarity/subjectivity | vaderSentiment, TextBlob    |
| Topics    | LDA, NMF topic modeling               | scikit-learn                |
| ML        | Ensemble (LR + SVM + RF + GB)         | scikit-learn                |
| Network   | Homophily index, graph analysis       | NetworkX                    |
| Toxicity  | Perspective API attributes            | Google Perspective API      |

### Machine Learning Performance

| Train Set         | Test Set | Accuracy                      |
| :---------------- | :------- | :---------------------------- |
| YouTube → YouTube | 75.1%    | High (repetitive slogans)     |
| Reddit → Reddit   | 62.5%    | Moderate (nuanced discourse)  |
| Reddit → YouTube  | 61.9%    | Cross-platform struggle       |
| YouTube → Reddit  | 57.2%    | Lowest (oversimplified model) |

## 📈 Visualizations

The analysis generates comprehensive visualizations including:

- Sentiment distribution comparisons
- Polarity vs. Subjectivity scatter plots
- Topic word clouds and frequency charts
- Echo chamber network graphs
- Toxicity heatmaps by stance
- Confusion matrices for ML models
- Feature importance plots

## 👥 Authors

- **Akshat Gupta**
- **Raghav Sarna**
- **Arsh Arora**
- **Mudasir Rasheed**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Google Perspective API for toxicity analysis

Reddit and YouTube for data access

The computational social science research community
