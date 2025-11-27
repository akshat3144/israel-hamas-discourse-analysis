# Phase 4: Advanced Analysis Results Summary

## Israel-Hamas War Discourse Analysis

**Date**: November 27, 2025

---

## 📊 Executive Summary

Phase 4 focused on advanced statistical modeling, structural analysis, and temporal patterns to deepen the understanding of discourse dynamics on Reddit and YouTube.

---

## Part A: Regression & Amplification (RQ2)

### 1. Predicting Engagement (Reddit)

We used OLS Regression to predict comment scores based on Stance and Sentiment.

**Model**: `Score ~ Sentiment (VADER) + Stance (Label)`

**Key Findings:**
- **Stance Matters**: 
  - **Neutral (N)** comments have significantly **lower scores** (-5.96) compared to Pro-Israel (baseline).
  - **Pro-Palestine (P)** comments have even **lower scores** (-8.60) compared to Pro-Israel.
  - This suggests **Pro-Israel content receives higher engagement (upvotes)** on the subreddits analyzed.
- **Sentiment Impact**: 
  - Sentiment score (`vader_compound`) was **not statistically significant** (p=0.305) in predicting score when controlling for stance.
  - Engagement is driven more by **"sides"** than by the emotional tone of the comment.

### 2. Algorithmic Amplification

We analyzed the impact of the `controversiality` flag on Reddit scores.

- **Non-Controversial Comments**: Average Score = **12.66**
- **Controversial Comments**: Average Score = **2.20**

**Interpretation**: Controversial comments (high mix of upvotes/downvotes) end up with lower *net* scores, indicating that while they generate reaction, the community consensus often penalizes them.

---

## Part B: Narrative Complexity (RQ3)

We used the **Flesch Reading Ease** score to measure narrative complexity.
*(Higher score = Easier to read, Lower score = More complex/academic)*

- **Reddit Mean Score**: **62.67** (Standard/Plain English)
- **YouTube Mean Score**: **65.72** (Slightly Easier/Simpler)

**Conclusion**: YouTube discourse is slightly more simplistic and accessible, while Reddit discourse is marginally more complex, aligning with its text-heavy, discussion-oriented nature.

---

## Part C: Structural & Temporal Analysis (RQ3)

### 1. Network Activity (Reddit)

- **Top Users**: A small group of users is highly active. The top user (`Enchilte`) posted **79 comments**.
- **Stance Consistency**: Active users (>5 comments) are most likely to be **Neutral (165 users)** or **Pro-Palestine (157 users)**, with fewer **Pro-Israel (92 users)** power users.
- This contrasts with the Regression finding where Pro-Israel comments got higher scores; fewer Pro-Israel users might be getting more upvotes per comment.

### 2. Response Time (Immediacy)

- **Median Response Time**: **360 minutes (6 hours)**.
- This indicates a relatively slow, asynchronous conversation pace on Reddit, allowing for more deliberative responses compared to real-time chat.

### 3. Conversation Volume (Thread Depth Proxy)

- **Reddit**: Avg **29.5 comments** per post.
- **YouTube**: Avg **21.5 comments** per video.
- **Conclusion**: Reddit threads tend to be deeper/longer, facilitating more sustained discussion than YouTube comment sections.

---

## 📁 Output Files

All visualizations and detailed results are saved in `advanced_analysis_output/`:

- **Regression**: `01_regression_coefficients.png`, `regression_results.txt`
- **Amplification**: `02_amplification_controversy.png`, `03_amplification_by_stance.png`
- **Complexity**: `04_complexity_platform_comparison.png`, `05_complexity_by_stance.png`
- **Network**: `06_network_top_users.png`
- **Temporal**: `07_response_time_distribution.png`, `08_response_time_by_stance.png`
- **Volume**: `09_conversation_volume.png`
