"""
Statistical Significance Testing for Sentiment Analysis
RQ1: Validate sentiment findings with chi-square, ANOVA, Kruskal-Wallis,
     Cramér's V, t-tests, and correlation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (chi2_contingency, f_oneway, ttest_ind,
                          pearsonr, spearmanr, kruskal, mannwhitneyu)
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = SENTIMENT_OUTPUT_DIR / 'statistical_tests'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

print("=" * 70)
print("STATISTICAL SIGNIFICANCE TESTING")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading sentiment-scored data...")
reddit_df  = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')
youtube_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'youtube_with_sentiment.csv')

reddit_df['Platform']  = 'Reddit'
youtube_df['Platform'] = 'YouTube'
combined_df = pd.concat([reddit_df, youtube_df], ignore_index=True)

print(f"✔ Reddit: {len(reddit_df):,}")
print(f"✔ YouTube: {len(youtube_df):,}")
print(f"✔ Combined: {len(combined_df):,}")

results_file = open(OUTPUT_DIR / 'statistical_test_results.txt', 'w', encoding='utf-8')
results_file.write("=" * 70 + "\n")
results_file.write("STATISTICAL SIGNIFICANCE TEST RESULTS\n")
results_file.write("RQ1: Sentiment Analysis Validation\n")
results_file.write("=" * 70 + "\n\n")

# ============================================================================
# HELPER: Cramér's V
# ============================================================================
def cramers_v(contingency_table):
    chi2, _, dof, _ = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / min(kcorr-1, rcorr-1)) if min(kcorr-1, rcorr-1) > 0 else 0

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var() + (n2-1)*g2.var()) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled if pooled != 0 else 0

def interp_d(d):
    d = abs(d)
    return 'Large' if d >= 0.8 else 'Medium' if d >= 0.5 else 'Small'

def interp_v(v):
    return 'Strong' if v >= 0.5 else 'Moderate' if v >= 0.3 else 'Weak'

# ============================================================================
# TEST 1: CHI-SQUARE + CRAMÉR'S V — Stance × Sentiment
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: CHI-SQUARE + CRAMÉR'S V — STANCE × SENTIMENT")
print("=" * 70)

results_file.write("TEST 1: CHI-SQUARE (Stance × Sentiment)\n" + "-" * 70 + "\n\n")

for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                        ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
    ct = pd.crosstab(df[lcol], df['vader_label'])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(ct)
    print(f"\n📊 {name}:")
    print(f"   Chi-square: {chi2:.4f}, df={dof}, p={p:.6f}, Cramér's V={v:.4f} ({interp_v(v)})")
    sig = p < 0.05
    results_file.write(f"{name}:\n")
    results_file.write(f"  Chi-square statistic: {chi2:.4f}\n")
    results_file.write(f"  P-value: {p:.6f}\n")
    results_file.write(f"  Degrees of freedom: {dof}\n")
    results_file.write(f"  Cramér's V: {v:.4f} ({interp_v(v)} association)\n")
    results_file.write(f"  Conclusion: {'REJECT H0 — stance and sentiment are DEPENDENT' if sig else 'FAIL TO REJECT H0'}\n\n")

# ============================================================================
# TEST 2: ONE-WAY ANOVA + KRUSKAL-WALLIS — Compound × Stance
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: ANOVA + KRUSKAL-WALLIS — COMPOUND SCORE × STANCE")
print("=" * 70)

results_file.write("=" * 70 + "\nTEST 2: ANOVA + KRUSKAL-WALLIS (Compound × Stance)\n" + "-" * 70 + "\n\n")

for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                        ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
    groups = [df[df[lcol] == s]['vader_compound'].dropna() for s in ['P', 'I', 'N']]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        f_stat, p_anova = f_oneway(*groups)
        h_stat, p_kw    = kruskal(*groups)
        print(f"\n📊 {name}:")
        print(f"   ANOVA:          F={f_stat:.4f}, p={p_anova:.6f}")
        print(f"   Kruskal-Wallis: H={h_stat:.4f}, p={p_kw:.6f}")
        results_file.write(f"{name}:\n")
        results_file.write(f"  ANOVA F: {f_stat:.4f}, p={p_anova:.6f}\n")
        results_file.write(f"  Kruskal-Wallis H: {h_stat:.4f}, p={p_kw:.6f}\n")
        results_file.write(f"  Conclusion (ANOVA): {'REJECT H0' if p_anova < 0.05 else 'FAIL TO REJECT H0'}\n")
        results_file.write(f"  Conclusion (K-W):   {'REJECT H0' if p_kw < 0.05 else 'FAIL TO REJECT H0'}\n\n")

# ============================================================================
# TEST 3: INDEPENDENT T-TESTS — Platform Differences
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: T-TESTS + MANN-WHITNEY U — PLATFORM DIFFERENCES")
print("=" * 70)

results_file.write("=" * 70 + "\nTEST 3: T-TESTS + MANN-WHITNEY U (Platform Differences)\n" + "-" * 70 + "\n\n")

comparisons = [
    ('Overall', reddit_df['vader_compound'].dropna(),
     youtube_df['vader_compound'].dropna()),
    ('Pro-Palestine',
     reddit_df[reddit_df[REDDIT_LABEL_COL]  == 'P']['vader_compound'].dropna(),
     youtube_df[youtube_df[YOUTUBE_LABEL_COL] == 'P']['vader_compound'].dropna()),
    ('Pro-Israel',
     reddit_df[reddit_df[REDDIT_LABEL_COL]  == 'I']['vader_compound'].dropna(),
     youtube_df[youtube_df[YOUTUBE_LABEL_COL] == 'I']['vader_compound'].dropna()),
    ('Neutral',
     reddit_df[reddit_df[REDDIT_LABEL_COL]  == 'N']['vader_compound'].dropna(),
     youtube_df[youtube_df[YOUTUBE_LABEL_COL] == 'N']['vader_compound'].dropna()),
]

t_results = []
for label, g1, g2 in comparisons:
    if len(g1) > 1 and len(g2) > 1:
        t_stat, p_t  = ttest_ind(g1, g2)
        u_stat, p_mw = mannwhitneyu(g1, g2, alternative='two-sided')
        d = cohens_d(g1, g2)
        print(f"\n📊 {label}: Reddit μ={g1.mean():.4f}, YouTube μ={g2.mean():.4f}")
        print(f"   t={t_stat:.4f}, p={p_t:.6f} | MWU p={p_mw:.6f} | d={d:.4f} ({interp_d(d)})")
        t_results.append({'Comparison': label,
                          'Reddit_mean': g1.mean(), 'YouTube_mean': g2.mean(),
                          't_stat': t_stat, 'p_ttest': p_t,
                          'u_stat': u_stat, 'p_mwu': p_mw,
                          'cohens_d': d, 'effect_size': interp_d(d),
                          'significant': p_t < 0.05})
        results_file.write(f"{label}:\n")
        results_file.write(f"  Reddit mean={g1.mean():.4f}, YouTube mean={g2.mean():.4f}\n")
        results_file.write(f"  t={t_stat:.4f}, p={p_t:.6f}\n")
        results_file.write(f"  Mann-Whitney U={u_stat:.0f}, p={p_mw:.6f}\n")
        results_file.write(f"  Cohen's d={d:.4f} ({interp_d(d)} effect)\n")
        results_file.write(f"  Conclusion: {'REJECT H0' if p_t < 0.05 else 'FAIL TO REJECT H0'}\n\n")

# ============================================================================
# TEST 4: CORRELATION — Sentiment × Engagement
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: CORRELATION — SENTIMENT × ENGAGEMENT")
print("=" * 70)

results_file.write("=" * 70 + "\nTEST 4: CORRELATION (Sentiment × Engagement)\n" + "-" * 70 + "\n\n")

# Reddit: vader_compound × score
reddit_df['score'] = pd.to_numeric(reddit_df['score'], errors='coerce')
valid_r = reddit_df[['vader_compound', 'score']].dropna()
if len(valid_r) > 1:
    rp, pp = pearsonr(valid_r['vader_compound'], valid_r['score'])
    rs, ps = spearmanr(valid_r['vader_compound'], valid_r['score'])
    print(f"\n📊 Reddit — Compound × Score: Pearson r={rp:.4f}(p={pp:.6f}), Spearman ρ={rs:.4f}(p={ps:.6f})")
    results_file.write(f"Reddit (Compound × Score):\n  Pearson r={rp:.4f}, p={pp:.6f}\n  Spearman ρ={rs:.4f}, p={ps:.6f}\n\n")
else:
    rp, pp, rs, ps = 0, 1, 0, 1; valid_r = pd.DataFrame()

# YouTube: vader_compound × likeCount
if 'likeCount' in youtube_df.columns:
    youtube_df['likeCount'] = pd.to_numeric(youtube_df['likeCount'], errors='coerce')
    valid_y = youtube_df[['vader_compound', 'likeCount']].dropna()
    if len(valid_y) > 1:
        ryp, pyp = pearsonr(valid_y['vader_compound'], valid_y['likeCount'])
        rys, pys = spearmanr(valid_y['vader_compound'], valid_y['likeCount'])
        print(f"📊 YouTube — Compound × Likes: Pearson r={ryp:.4f}(p={pyp:.6f}), Spearman ρ={rys:.4f}(p={pys:.6f})")
        results_file.write(f"YouTube (Compound × likeCount):\n  Pearson r={ryp:.4f}, p={pyp:.6f}\n  Spearman ρ={rys:.4f}, p={pys:.6f}\n\n")
    else:
        valid_y = pd.DataFrame()
else:
    valid_y = pd.DataFrame()

# ============================================================================
# TEST 5: SUBJECTIVITY ACROSS STANCES — ANOVA + K-W
# ============================================================================
print("\n" + "=" * 70)
print("TEST 5: SUBJECTIVITY × STANCE (ANOVA + KRUSKAL-WALLIS)")
print("=" * 70)

results_file.write("=" * 70 + "\nTEST 5: SUBJECTIVITY × STANCE\n" + "-" * 70 + "\n\n")

for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                        ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
    groups = [df[df[lcol] == s]['textblob_subjectivity'].dropna() for s in ['P', 'I', 'N']]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        f_stat, p_anova = f_oneway(*groups)
        h_stat, p_kw    = kruskal(*groups)
        print(f"\n📊 {name} Subjectivity: F={f_stat:.4f}(p={p_anova:.6f}), H={h_stat:.4f}(p={p_kw:.6f})")
        results_file.write(f"{name}:\n  F={f_stat:.4f}, p={p_anova:.6f}\n  H={h_stat:.4f}, p_kw={p_kw:.6f}\n")
        for s, n_ in [('P', 'Pro-Palestine'), ('I', 'Pro-Israel'), ('N', 'Neutral')]:
            m = df[df[lcol] == s]['textblob_subjectivity'].mean()
            results_file.write(f"    {n_}: {m:.4f}\n")
        results_file.write("\n")

# ============================================================================
# TEST 6: EFFECT SIZES SUMMARY
# ============================================================================
results_file.write("=" * 70 + "\nTEST 6: EFFECT SIZES (COHEN'S D)\n" + "-" * 70 + "\n\n")
for row in t_results:
    results_file.write(f"  {row['Comparison']}: d={row['cohens_d']:.4f} ({row['effect_size']})\n")
results_file.write("\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# --- V1: Chi-square contingency heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Stance × Sentiment Contingency Tables', fontsize=14, fontweight='bold')
for ax, df, lcol, title, cmap in [
        (axes[0], reddit_df, REDDIT_LABEL_COL,  'Reddit', 'Blues'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube', 'Reds')]:
    ct = pd.crosstab(df[lcol], df['vader_label'])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(ct)
    sns.heatmap(ct, annot=True, fmt='d', cmap=cmap, ax=ax)
    ax.set_title(f'{title}\nχ²={chi2:.2f}, p={p:.4f}, V={v:.3f}', fontweight='bold')
    ax.set_xlabel('Sentiment (VADER)'); ax.set_ylabel('Stance')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_chi_square_contingency_tables.png', dpi=300, bbox_inches='tight')
print("\n✔ Saved: 01_chi_square_contingency_tables.png")
plt.close()

# --- V2: Sentiment × Engagement scatter ---
n_plots = (1 if len(valid_r) == 0 or len(valid_y) == 0 else 2)
fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
if n_plots == 1:
    axes = [axes]

if len(valid_r) > 0:
    axes[0].scatter(valid_r['vader_compound'], valid_r['score'],
                    alpha=0.15, s=10, color='#3498db')
    axes[0].set_xlabel('VADER Compound'); axes[0].set_ylabel('Reddit Score')
    axes[0].set_title(f'Reddit: r={rp:.3f}, p={pp:.4f}', fontweight='bold')
    axes[0].grid(alpha=0.3)

if len(valid_y) > 0 and n_plots == 2:
    axes[1].scatter(valid_y['vader_compound'], valid_y['likeCount'],
                    alpha=0.15, s=10, color='#e74c3c')
    axes[1].set_xlabel('VADER Compound'); axes[1].set_ylabel('YouTube Likes')
    axes[1].set_title(f'YouTube: r={ryp:.3f}, p={pyp:.4f}', fontweight='bold')
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_sentiment_engagement_correlation.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 02_sentiment_engagement_correlation.png")
plt.close()

# --- V3: Summary table visualization ---
if t_results:
    summary_df = pd.DataFrame(t_results)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_bar = ['#27ae60' if s else '#e74c3c' for s in summary_df['significant']]
    y_pos = np.arange(len(summary_df))
    ax.barh(y_pos, [1] * len(summary_df), color=colors_bar, alpha=0.35)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df['Comparison'])
    for i, row in summary_df.iterrows():
        mark = '✔ Sig.' if row['significant'] else '✗ N.S.'
        ax.text(0.5, i, f"{mark}  d={row['cohens_d']:.3f} ({row['effect_size']})",
                ha='center', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_title('Platform Sentiment Differences — T-test Results (α = 0.05)',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_statistical_tests_summary.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 03_statistical_tests_summary.png")
    plt.close()

results_file.close()
print("✔ Saved: statistical_test_results.txt")

print("\n" + "=" * 70)
print("✅ STATISTICAL TESTING COMPLETE")
print("=" * 70)
