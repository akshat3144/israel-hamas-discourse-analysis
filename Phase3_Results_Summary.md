# Phase 3: Sentiment Analysis & Topic Modeling - Results Summary

## Israel-Hamas War Discourse Analysis

**Date**: November 27, 2025

---

## 📊 Executive Summary

Phase 3 completed comprehensive sentiment analysis and topic modeling on Reddit (9,973 posts) and YouTube (9,389 comments) to understand discourse patterns around the Israel-Hamas conflict.

---

## Part A: Sentiment Analysis Results

### 1. Overall Sentiment Distribution

#### Reddit (VADER):

- **Negative**: 45.1% (4,499 posts)
- **Positive**: 33.4% (3,326 posts)
- **Neutral**: 21.5% (2,148 posts)

#### YouTube (VADER):

- **Positive**: 44.8% (4,204 comments)
- **Negative**: 32.5% (3,056 comments)
- **Neutral**: 22.7% (2,129 comments)

### 🔍 Key Finding #1: Platform Sentiment Differences

- **Reddit skews NEGATIVE** (45.1% negative)
- **YouTube skews POSITIVE** (44.8% positive)
- This suggests different discourse tones: Reddit more critical/analytical, YouTube more supportive/reactive

---

### 2. Sentiment by Stance Analysis

#### Reddit Sentiment by Stance (VADER Compound Score):

| Stance            | Mean   | Median | Interpretation      |
| ----------------- | ------ | ------ | ------------------- |
| Pro-Israel (I)    | -0.225 | -0.285 | **Most negative**   |
| Pro-Palestine (P) | -0.120 | 0.000  | Moderately negative |
| Neutral (N)       | -0.036 | 0.000  | Nearly balanced     |

#### YouTube Sentiment by Stance (VADER Compound Score):

| Stance            | Mean  | Median | Interpretation    |
| ----------------- | ----- | ------ | ----------------- |
| Pro-Palestine (P) | 0.106 | 0.000  | **Most positive** |
| Neutral (N)       | 0.090 | 0.000  | Slightly positive |
| Pro-Israel (I)    | 0.012 | 0.000  | Nearly neutral    |

### 🔍 Key Finding #2: Stance-Sentiment Correlation

- **Reddit Pro-Israel content is most negative** (-0.225)
- **YouTube Pro-Palestine content is most positive** (0.106)
- Platform culture influences how stances are expressed

---

### 3. Subjectivity Analysis (TextBlob)

#### Reddit Average Subjectivity by Stance:

- Pro-Israel (I): 0.405 (Most subjective)
- Pro-Palestine (P): 0.393
- Neutral (N): 0.374 (Most objective)

#### YouTube Average Subjectivity by Stance:

- Pro-Palestine (P): 0.406 (Most subjective)
- Pro-Israel (I): 0.319
- Neutral (N): 0.291 (Most objective)

### 🔍 Key Finding #3: Emotional Expression

- **Pro-Palestine content more subjective on YouTube** (0.406)
- **Pro-Israel content more subjective on Reddit** (0.405)
- Neutral stances maintain objectivity across platforms

---

## Part B: Topic Modeling Results

### 1. Most Frequent Words (excluding common terms)

#### Reddit Top 10 Keywords:

1. **don't** (1,435)
2. **palestinians** (1,302)
3. **jews** (1,158)
4. **i'm** (768)
5. **right** (740)
6. **that's** (723)
7. **state** (701)
8. **land** (676)
9. **country** (612)
10. **did** (595)

#### YouTube Top 10 Keywords:

1. **free** (827) ← Notable difference
2. **don't** (554)
3. **allah** (529) ← Religious context
4. **world** (438)
5. **palestinians** (427)
6. **god** (387) ← Religious context
7. **jews** (322)
8. **support** (320)
9. **country** (265)
10. **children** (253)

### 🔍 Key Finding #4: Narrative Framing

- **YouTube**: "free," "allah," "god," "support" → Emotional/religious framing
- **Reddit**: "state," "land," "country," "right" → Political/territorial framing
- YouTube more action-oriented ("free," "support")
- Reddit more analytical ("state," "land," "right")

---

### 2. LDA Topic Modeling (5 Topics per Platform)

#### Reddit Key Topics:

1. **Topic 1**: Personal opinions/expressions ("people," "like," "I'm," "don't")
2. **Topic 2**: Israel-Palestine conflict basics ("israel," "palestinians," "country," "land")
3. **Topic 3**: Jewish-Arab relations ("jews," "arab," "jewish," "arabs," "state")
4. **Topic 4**: Hamas and humanitarian crisis ("hamas," "children," "civilians," "military," "aid")
5. **Topic 5**: Gaza situation and genocide discourse ("gaza," "genocide," "west," "state")

#### YouTube Key Topics:

1. **Topic 1**: Support movements ("free palestine," "support," "stand," "love")
2. **Topic 2**: Religious appeals ("allah," "god," "bless," "muslim," "islam")
3. **Topic 3**: Anti-war sentiment ("stop," "genocide," "killing")
4. **Topic 4**: Media and perception ("media," "think," "know," "say")
5. **Topic 5**: Hamas-Gaza situation ("hamas," "gaza," "india," "state")

### 🔍 Key Finding #5: Thematic Differences

- **Reddit**: More focused on historical context, territorial disputes, civilian impact
- **YouTube**: More focused on solidarity movements, religious framing, emotional appeals
- Reddit engages with complexity (state, land, rights)
- YouTube emphasizes moral positions (free, support, stop)

---

### 3. Words by Stance (Reddit)

#### Pro-Palestine Top Words:

- **Palestinians** (663)
- **Don't** (578)
- **Jews** (481)
- **Genocide** (273) ← Unique to P stance
- **State** (356)

#### Pro-Israel Top Words:

- **Palestinians** (373)
- **Jews** (371)
- **Don't** (273)
- **Land** (205)
- **Arab** (161)
- **Civilians** (147)

#### Neutral Top Words:

- **Don't** (581)
- **I'm** (374)
- **Jews** (306)
- **Right** (271)
- **Country** (242)

### 🔍 Key Finding #6: Stance-Specific Narratives

- **Pro-Palestine**: Emphasizes "genocide" frame
- **Pro-Israel**: Focus on "civilians," "arab," historical land claims
- **Neutral**: More questioning ("don't," "I'm," "right")

---

## 📈 Research Questions Answered

### RQ1: How are narratives and sentiments represented across platforms?

**Answer:**

1. **Sentiment Polarity**: Reddit more negative (45%), YouTube more positive (45%)
2. **Narrative Framing**:
   - Reddit: Analytical, territorial, historical
   - YouTube: Emotional, religious, solidarity-based
3. **Stance-Sentiment Link**:
   - Pro-Israel content more negative on Reddit
   - Pro-Palestine content more positive on YouTube

### RQ3: Platform Differences in Framing

**Answer:**

1. **Reddit (Discussion-based)**:

   - Longer content (median 132 chars)
   - Political/territorial framing ("state," "land," "right")
   - Analytical topics (Jewish-Arab relations, civilian impact)
   - More balanced sentiment distribution

2. **YouTube (Reactive-based)**:
   - Shorter content (median 54 chars)
   - Emotional/religious framing ("free," "allah," "support")
   - Action-oriented language
   - More polarized positive sentiment

---

## 🎯 Critical Insights

### 1. Platform Affordances Shape Discourse

- **Reddit's threaded discussions** → More nuanced, analytical debate
- **YouTube's comment system** → More reactive, emotional expressions

### 2. Sentiment-Stance Misalignment

- Not all Pro-Palestine content is negative (YouTube: +0.106)
- Not all Pro-Israel content is positive (Reddit: -0.225)
- Context and platform matter more than stance alone

### 3. Religious vs. Political Framing

- **YouTube**: Heavy religious framing ("allah," "god," "bless")
- **Reddit**: Political/legal framing ("state," "land," "rights")
- Different audiences, different discourse modes

### 4. Vocabulary Divergence

- **"Free"** appears 827 times on YouTube, minimal on Reddit
- **"Genocide"** more prominent in Pro-Palestine Reddit discourse
- Platform-specific keywords reveal different priorities

---

## 📁 Deliverables Generated

### Sentiment Analysis:

1. `01_sentiment_distribution.png` - Overall sentiment by platform
2. `02_sentiment_by_stance_heatmap.png` - Stance-sentiment correlation
3. `03_compound_score_by_stance.png` - Detailed sentiment scores
4. `04_polarity_subjectivity_scatter.png` - Emotional expression patterns
5. `05_platform_sentiment_comparison.png` - Reddit vs YouTube
6. `reddit_with_sentiment.csv` - Enhanced dataset
7. `youtube_with_sentiment.csv` - Enhanced dataset
8. `sentiment_summary_report.txt` - Statistical summary

### Topic Modeling:

1. `01_word_frequency.png` - Top words by platform
2. `02_lda_topic_heatmaps.png` - Topic distributions
3. `topic_modeling_report.txt` - Full topic analysis

### Interactive Analysis:

- `Phase3_Sentiment_TopicModeling.ipynb` - Jupyter notebook for exploration

---

## 🔬 Methodological Notes

### Sentiment Analysis:

- **VADER**: Optimized for social media text, captures intensity
- **TextBlob**: Measures polarity (-1 to +1) and subjectivity (0 to 1)
- Both methods used for validation and comparison

### Topic Modeling:

- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic discovery
- **NMF (Non-negative Matrix Factorization)**: TF-IDF weighted topics
- 5 topics per platform for interpretability

---

## 📊 Statistical Highlights

| Metric                     | Reddit             | YouTube               |
| -------------------------- | ------------------ | --------------------- |
| **Total Analyzed**         | 9,973              | 9,389                 |
| **Dominant Sentiment**     | Negative (45.1%)   | Positive (44.8%)      |
| **Most Subjective Stance** | Pro-Israel (0.405) | Pro-Palestine (0.406) |
| **Avg Compound (VADER)**   | -0.127             | 0.069                 |
| **Top Keyword**            | "don't"            | "free"                |
| **Religious Terms Freq**   | Low                | High                  |

---

## 🚀 Next Steps (Phase 4)

1. **Statistical Significance Testing**

   - Chi-square for sentiment-stance independence
   - ANOVA for engagement-sentiment correlation
   - T-tests for platform differences

2. **Engagement Analysis (RQ2)**

   - Correlation: Sentiment × Upvotes/Likes
   - Viral content analysis
   - Algorithmic visibility patterns

3. **Temporal Analysis**

   - Sentiment trends over time
   - Event-driven discourse shifts

4. **Network Analysis**
   - User interaction patterns
   - Echo chamber detection

---

## 📝 Limitations

1. **Context Loss**: Sentiment analysis may miss sarcasm/irony
2. **Label Validation**: Topics require manual interpretation
3. **Cross-platform Comparison**: Different user demographics
4. **Temporal Bias**: Snapshot of specific time period

---

## ✅ Conclusion

Phase 3 successfully identified **clear platform-specific discourse patterns**:

- **Reddit**: Critical, analytical, politically-framed discussions with more negative sentiment
- **YouTube**: Emotional, religiously-framed, supportive comments with more positive sentiment
- **Sentiment-Stance Correlation**: Platform culture significantly influences how positions are expressed
- **Narrative Diversity**: 5 distinct topics per platform reveal multifaceted discourse

These findings directly address RQ1 (narrative representation) and RQ3 (platform differences), setting the foundation for engagement analysis in Phase 4.

---

**Report Generated**: November 27, 2025  
**Analysis Tools**: VADER, TextBlob, LDA, NMF, scikit-learn  
**Total Data Points**: 19,362 (Reddit: 9,973 | YouTube: 9,389)
