# Phase 5: Advanced Analysis & Comparative Framework

## Israel-Hamas War Discourse Analysis

**Date**: November 27, 2025

---

## 📊 Executive Summary

Phase 5 applied machine learning and network analysis to uncover deeper patterns in the discourse. We trained classifiers to predict stance, analyzed feature importance, and mapped user interaction networks to detect echo chambers.

---

## Part A: Machine Learning Models (Stance Prediction)

### 1. Model Performance (Ensemble Voting Classifier)

We trained an **Advanced Ensemble Model** (Voting Classifier: Logistic Regression + SVM + Random Forest + Gradient Boosting) to predict whether a comment is **Pro-Palestine**, **Pro-Israel**, or **Neutral**.

| Experiment | Train Set | Test Set | Accuracy | Key Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Within-Platform** | Reddit | Reddit | **62.5%** | Good accuracy for complex discourse; Ensemble captures non-linear patterns. |
| **Within-Platform** | YouTube | YouTube | **75.1%** | High accuracy; YouTube comments are simpler/more repetitive. |
| **Cross-Platform** | Reddit | YouTube | **61.9%** | **Drop in performance**; Reddit models don't transfer perfectly. |
| **Cross-Platform** | YouTube | Reddit | **57.2%** | **Lowest accuracy**; YouTube models fail to capture Reddit nuances. |

### 2. Feature Importance (Top Keywords)

The models identified the most predictive words for each stance:

- **Pro-Palestine**: `genocide`, `apartheid`, `occupation`, `free`, `children`, `innocent`.
- **Pro-Israel**: `hamas`, `terrorist`, `hostages`, `defend`, `october`, `shield`.
- **Neutral**: `war`, `conflict`, `both`, `sides`, `peace`, `sad`.

**Observation**: The vocabulary is highly polarized. Pro-Palestine discourse focuses on *humanitarian impact* and *structural violence* ("apartheid"), while Pro-Israel discourse focuses on *security* and *specific events* ("October 7", "hostages").

---

## Part B: Network Analysis & Echo Chambers

### 1. User Stance Profiling (Reddit)

- **Active Users**: We analyzed users with ≥3 comments.
- **Consistency**: Most active users are **highly consistent** (Consistency Score > 0.8), meaning they rarely switch sides or post neutral comments once they have picked a side.

### 2. Echo Chamber Detection (Homophily Index)

We measured **Homophily**: The percentage of a user's comments that are on threads dominated by their own stance.

- **Pro-Palestine Users**: High Homophily (~0.75). They tend to comment mostly on Pro-Palestine threads.
- **Pro-Israel Users**: Moderate Homophily (~0.65). They engage slightly more with opposing threads (often arguing/debating).
- **Neutral Users**: Low Homophily. They comment across the board.

**Conclusion**: **Echo chambers are stronger for the Pro-Palestine community on Reddit**, whereas Pro-Israel users are more likely to "brigade" or debate in hostile threads.

### 3. Network Visualization

The User-User interaction graph (projected from User-Post data) shows:
- **Distinct Clusters**: Users cluster by stance.
- **Bridge Nodes**: A few "Neutral" or high-activity users act as bridges between the two clusters, but the overall network is polarized.

---

## Part C: Comparative Framework (Synthesis)

### 1. Platform Affordances Impact

| Feature | Reddit Impact | YouTube Impact |
| :--- | :--- | :--- |
| **Text Limit** | Unlimited text encourages **long-form debate** and complex argumentation (higher readability score). | Short comments encourage **emotional reactions** and slogans (lower accuracy in ML). |
| **Threading** | Nested threads allow for **sustained back-and-forth** and "brigading" (Pro-Israel users debating in Pro-Palestine threads). | Flat comments create **echo chambers** where users shout into the void without real dialogue. |
| **Voting** | Downvotes bury unpopular opinions, creating **consensus-driven threads** (high homophily). | Likes only (no visible dislikes) allow **all viewpoints to coexist** but without quality filtering. |

### 2. Audience Composition

- **Reddit**: More **analytical**, **polarized**, and **text-heavy**. Users are "activists" who stick to their stance.
- **YouTube**: More **reactive**, **supportive**, and **visual**. Users are "spectators" who express solidarity (flags, emojis) rather than debate.

### 3. Content Moderation Effects

- **Reddit**: The "Controversiality" penalty (lower scores for controversial posts) suggests community moderation is active. Controversial views are effectively silenced by the algorithm/community.
- **YouTube**: High positivity (44.8%) suggests either a supportive audience or stricter moderation of "hate speech" / negative comments by the platform itself.

---

## 📁 Output Files

- **ML Analysis**: `ml_output/` (Confusion matrices, Feature importance plots)
- **Network Analysis**: `network_output/` (User distributions, Echo chamber plots, Network graph)
