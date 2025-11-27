"""
Phase 3: Statistical Significance Testing
Validate findings with chi-square, ANOVA, t-tests, and correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, pearsonr, spearmanr
import os

# Create output directory
os.makedirs('statistical_tests_output', exist_ok=True)

print("="*70)
print("PHASE 3: STATISTICAL SIGNIFICANCE TESTING")
print("="*70)
print()

# ============================================================================
# 1. LOAD DATA WITH SENTIMENT SCORES
# ============================================================================
print("📂 Loading data with sentiment scores...")
reddit_df = pd.read_csv('sentiment_output/reddit_with_sentiment.csv')
youtube_df = pd.read_csv('sentiment_output/youtube_with_sentiment.csv')

# Add platform identifier
reddit_df['Platform'] = 'Reddit'
youtube_df['Platform'] = 'YouTube'

# Combine datasets
combined_df = pd.concat([reddit_df, youtube_df], ignore_index=True)

print(f"✓ Reddit: {len(reddit_df)} rows")
print(f"✓ YouTube: {len(youtube_df)} rows")
print(f"✓ Combined: {len(combined_df)} rows")
print()

# ============================================================================
# 2. CHI-SQUARE TEST: STANCE × SENTIMENT INDEPENDENCE
# ============================================================================
print("="*70)
print("TEST 1: CHI-SQUARE - STANCE × SENTIMENT INDEPENDENCE")
print("="*70)
print()

results_file = open('statistical_tests_output/statistical_test_results.txt', 'w', encoding='utf-8')
results_file.write("="*70 + "\n")
results_file.write("STATISTICAL SIGNIFICANCE TEST RESULTS\n")
results_file.write("Phase 3: Sentiment Analysis Validation\n")
results_file.write("="*70 + "\n\n")

# Reddit Chi-Square
print("📊 Reddit: Stance × Sentiment")
reddit_contingency = pd.crosstab(reddit_df['Label'], reddit_df['vader_label'])
chi2_reddit, p_reddit, dof_reddit, expected_reddit = chi2_contingency(reddit_contingency)

print(f"   Chi-square statistic: {chi2_reddit:.4f}")
print(f"   P-value: {p_reddit:.6f}")
print(f"   Degrees of freedom: {dof_reddit}")
print(f"   Result: {'SIGNIFICANT' if p_reddit < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("TEST 1: CHI-SQUARE TEST\n")
results_file.write("Hypothesis: Stance and Sentiment are independent\n")
results_file.write("-"*70 + "\n\n")
results_file.write("REDDIT:\n")
results_file.write(f"  Chi-square statistic: {chi2_reddit:.4f}\n")
results_file.write(f"  P-value: {p_reddit:.6f}\n")
results_file.write(f"  Degrees of freedom: {dof_reddit}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Stance and Sentiment are DEPENDENT' if p_reddit < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# YouTube Chi-Square
print("📊 YouTube: Stance × Sentiment")
label_col_youtube = 'label' if 'label' in youtube_df.columns else 'Label'
youtube_contingency = pd.crosstab(youtube_df[label_col_youtube], youtube_df['vader_label'])
chi2_youtube, p_youtube, dof_youtube, expected_youtube = chi2_contingency(youtube_contingency)

print(f"   Chi-square statistic: {chi2_youtube:.4f}")
print(f"   P-value: {p_youtube:.6f}")
print(f"   Degrees of freedom: {dof_youtube}")
print(f"   Result: {'SIGNIFICANT' if p_youtube < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("YOUTUBE:\n")
results_file.write(f"  Chi-square statistic: {chi2_youtube:.4f}\n")
results_file.write(f"  P-value: {p_youtube:.6f}\n")
results_file.write(f"  Degrees of freedom: {dof_youtube}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Stance and Sentiment are DEPENDENT' if p_youtube < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# Visualize contingency tables
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Stance × Sentiment Contingency Tables', fontsize=14, fontweight='bold')

sns.heatmap(reddit_contingency, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title(f'Reddit (χ² = {chi2_reddit:.2f}, p = {p_reddit:.4f})', fontweight='bold')
axes[0].set_xlabel('Sentiment (VADER)')
axes[0].set_ylabel('Stance')

sns.heatmap(youtube_contingency, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title(f'YouTube (χ² = {chi2_youtube:.2f}, p = {p_youtube:.4f})', fontweight='bold')
axes[1].set_xlabel('Sentiment (VADER)')
axes[1].set_ylabel('Stance')

plt.tight_layout()
plt.savefig('statistical_tests_output/01_chi_square_contingency_tables.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_chi_square_contingency_tables.png")
print()

# ============================================================================
# 3. ANOVA: SENTIMENT SCORES ACROSS STANCES
# ============================================================================
print("="*70)
print("TEST 2: ONE-WAY ANOVA - SENTIMENT SCORES ACROSS STANCES")
print("="*70)
print()

results_file.write("="*70 + "\n")
results_file.write("TEST 2: ONE-WAY ANOVA\n")
results_file.write("Hypothesis: Mean sentiment scores are equal across all stances\n")
results_file.write("-"*70 + "\n\n")

# Reddit ANOVA
print("📊 Reddit: Compound Score × Stance")
reddit_groups = [reddit_df[reddit_df['Label'] == stance]['vader_compound'].dropna() 
                 for stance in ['P', 'I', 'N']]
f_stat_reddit, p_anova_reddit = f_oneway(*reddit_groups)

print(f"   F-statistic: {f_stat_reddit:.4f}")
print(f"   P-value: {p_anova_reddit:.6f}")
print(f"   Result: {'SIGNIFICANT' if p_anova_reddit < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("REDDIT (VADER Compound Score by Stance):\n")
results_file.write(f"  F-statistic: {f_stat_reddit:.4f}\n")
results_file.write(f"  P-value: {p_anova_reddit:.6f}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Mean sentiment scores DIFFER significantly across stances' if p_anova_reddit < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# YouTube ANOVA
print("📊 YouTube: Compound Score × Stance")
label_col_youtube = 'label' if 'label' in youtube_df.columns else 'Label'
youtube_groups = [youtube_df[youtube_df[label_col_youtube] == stance]['vader_compound'].dropna() 
                  for stance in ['P', 'I', 'N']]
f_stat_youtube, p_anova_youtube = f_oneway(*youtube_groups)

print(f"   F-statistic: {f_stat_youtube:.4f}")
print(f"   P-value: {p_anova_youtube:.6f}")
print(f"   Result: {'SIGNIFICANT' if p_anova_youtube < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("YOUTUBE (VADER Compound Score by Stance):\n")
results_file.write(f"  F-statistic: {f_stat_youtube:.4f}\n")
results_file.write(f"  P-value: {p_anova_youtube:.6f}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Mean sentiment scores DIFFER significantly across stances' if p_anova_youtube < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# ============================================================================
# 4. T-TESTS: PLATFORM DIFFERENCES
# ============================================================================
print("="*70)
print("TEST 3: INDEPENDENT T-TESTS - PLATFORM DIFFERENCES")
print("="*70)
print()

results_file.write("="*70 + "\n")
results_file.write("TEST 3: INDEPENDENT T-TESTS\n")
results_file.write("Hypothesis: Mean sentiment scores are equal between Reddit and YouTube\n")
results_file.write("-"*70 + "\n\n")

# Overall sentiment comparison
print("📊 Overall Sentiment: Reddit vs YouTube")
t_stat_overall, p_ttest_overall = ttest_ind(
    reddit_df['vader_compound'].dropna(),
    youtube_df['vader_compound'].dropna()
)

reddit_mean = reddit_df['vader_compound'].mean()
youtube_mean = youtube_df['vader_compound'].mean()

print(f"   Reddit mean: {reddit_mean:.4f}")
print(f"   YouTube mean: {youtube_mean:.4f}")
print(f"   T-statistic: {t_stat_overall:.4f}")
print(f"   P-value: {p_ttest_overall:.6f}")
print(f"   Result: {'SIGNIFICANT' if p_ttest_overall < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("OVERALL SENTIMENT (Reddit vs YouTube):\n")
results_file.write(f"  Reddit mean compound score: {reddit_mean:.4f}\n")
results_file.write(f"  YouTube mean compound score: {youtube_mean:.4f}\n")
results_file.write(f"  T-statistic: {t_stat_overall:.4f}\n")
results_file.write(f"  P-value: {p_ttest_overall:.6f}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Platforms have DIFFERENT mean sentiments' if p_ttest_overall < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# Pro-Palestine stance comparison
print("📊 Pro-Palestine Stance: Reddit vs YouTube")
label_col_youtube = 'label' if 'label' in youtube_df.columns else 'Label'
reddit_p = reddit_df[reddit_df['Label'] == 'P']['vader_compound'].dropna()
youtube_p = youtube_df[youtube_df[label_col_youtube] == 'P']['vader_compound'].dropna()
t_stat_p, p_ttest_p = ttest_ind(reddit_p, youtube_p)

print(f"   Reddit mean: {reddit_p.mean():.4f}")
print(f"   YouTube mean: {youtube_p.mean():.4f}")
print(f"   T-statistic: {t_stat_p:.4f}")
print(f"   P-value: {p_ttest_p:.6f}")
print(f"   Result: {'SIGNIFICANT' if p_ttest_p < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("PRO-PALESTINE STANCE (Reddit vs YouTube):\n")
results_file.write(f"  Reddit mean: {reddit_p.mean():.4f}\n")
results_file.write(f"  YouTube mean: {youtube_p.mean():.4f}\n")
results_file.write(f"  T-statistic: {t_stat_p:.4f}\n")
results_file.write(f"  P-value: {p_ttest_p:.6f}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Platform difference is SIGNIFICANT' if p_ttest_p < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# Pro-Israel stance comparison
print("📊 Pro-Israel Stance: Reddit vs YouTube")
label_col_youtube = 'label' if 'label' in youtube_df.columns else 'Label'
reddit_i = reddit_df[reddit_df['Label'] == 'I']['vader_compound'].dropna()
youtube_i = youtube_df[youtube_df[label_col_youtube] == 'I']['vader_compound'].dropna()
t_stat_i, p_ttest_i = ttest_ind(reddit_i, youtube_i)

print(f"   Reddit mean: {reddit_i.mean():.4f}")
print(f"   YouTube mean: {youtube_i.mean():.4f}")
print(f"   T-statistic: {t_stat_i:.4f}")
print(f"   P-value: {p_ttest_i:.6f}")
print(f"   Result: {'SIGNIFICANT' if p_ttest_i < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("PRO-ISRAEL STANCE (Reddit vs YouTube):\n")
results_file.write(f"  Reddit mean: {reddit_i.mean():.4f}\n")
results_file.write(f"  YouTube mean: {youtube_i.mean():.4f}\n")
results_file.write(f"  T-statistic: {t_stat_i:.4f}\n")
results_file.write(f"  P-value: {p_ttest_i:.6f}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Platform difference is SIGNIFICANT' if p_ttest_i < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# ============================================================================
# 5. CORRELATION ANALYSIS: SENTIMENT × ENGAGEMENT
# ============================================================================
print("="*70)
print("TEST 4: CORRELATION - SENTIMENT × ENGAGEMENT METRICS")
print("="*70)
print()

results_file.write("="*70 + "\n")
results_file.write("TEST 4: CORRELATION ANALYSIS\n")
results_file.write("Hypothesis: Sentiment correlates with engagement metrics\n")
results_file.write("-"*70 + "\n\n")

# Reddit: Sentiment × Score
reddit_df['score'] = pd.to_numeric(reddit_df['score'], errors='coerce')
valid_reddit = reddit_df[['vader_compound', 'score']].dropna()

if len(valid_reddit) > 0:
    corr_reddit_pearson, p_corr_reddit_pearson = pearsonr(valid_reddit['vader_compound'], valid_reddit['score'])
    corr_reddit_spearman, p_corr_reddit_spearman = spearmanr(valid_reddit['vader_compound'], valid_reddit['score'])
    
    print("📊 Reddit: Sentiment × Score (Upvotes)")
    print(f"   Pearson correlation: {corr_reddit_pearson:.4f} (p = {p_corr_reddit_pearson:.6f})")
    print(f"   Spearman correlation: {corr_reddit_spearman:.4f} (p = {p_corr_reddit_spearman:.6f})")
    print(f"   Result: {'SIGNIFICANT' if p_corr_reddit_pearson < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
    print()
    
    results_file.write("REDDIT (Sentiment × Score):\n")
    results_file.write(f"  Pearson r: {corr_reddit_pearson:.4f} (p = {p_corr_reddit_pearson:.6f})\n")
    results_file.write(f"  Spearman ρ: {corr_reddit_spearman:.4f} (p = {p_corr_reddit_spearman:.6f})\n")
    results_file.write(f"  Conclusion: {'SIGNIFICANT correlation - Sentiment affects engagement' if p_corr_reddit_pearson < 0.05 else 'NO significant correlation'}\n\n")

# YouTube: Sentiment × Engagement (if available)
engagement_col = None
for col in ['likeCount', 'likes', 'like_count']:
    if col in youtube_df.columns:
        engagement_col = col
        break

if engagement_col:
    youtube_df[engagement_col] = pd.to_numeric(youtube_df[engagement_col], errors='coerce')
    valid_youtube = youtube_df[['vader_compound', engagement_col]].dropna()

    if len(valid_youtube) > 0:
        corr_youtube_pearson, p_corr_youtube_pearson = pearsonr(valid_youtube['vader_compound'], valid_youtube[engagement_col])
        corr_youtube_spearman, p_corr_youtube_spearman = spearmanr(valid_youtube['vader_compound'], valid_youtube[engagement_col])
        
        print(f"📊 YouTube: Sentiment × {engagement_col}")
        print(f"   Pearson correlation: {corr_youtube_pearson:.4f} (p = {p_corr_youtube_pearson:.6f})")
        print(f"   Spearman correlation: {corr_youtube_spearman:.4f} (p = {p_corr_youtube_spearman:.6f})")
        print(f"   Result: {'SIGNIFICANT' if p_corr_youtube_pearson < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
        print()
        
        results_file.write(f"YOUTUBE (Sentiment × {engagement_col}):\n")
        results_file.write(f"  Pearson r: {corr_youtube_pearson:.4f} (p = {p_corr_youtube_pearson:.6f})\n")
        results_file.write(f"  Spearman ρ: {corr_youtube_spearman:.4f} (p = {p_corr_youtube_spearman:.6f})\n")
        results_file.write(f"  Conclusion: {'SIGNIFICANT correlation - Sentiment affects engagement' if p_corr_youtube_pearson < 0.05 else 'NO significant correlation'}\n\n")
else:
    print("📊 YouTube: No engagement metrics available")
    print()
    results_file.write("YOUTUBE: No engagement metrics available in dataset\n\n")
    corr_youtube_pearson, p_corr_youtube_pearson = 0, 1  # Default values for summary
    valid_youtube = pd.DataFrame()  # Empty for visualization

# Visualize correlations
if len(valid_youtube) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sentiment × Engagement Correlations', fontsize=14, fontweight='bold')

    # Reddit scatter
    axes[0].scatter(valid_reddit['vader_compound'], valid_reddit['score'], alpha=0.3, s=20, color='#3498db')
    axes[0].set_xlabel('VADER Compound Score')
    axes[0].set_ylabel('Reddit Score (Upvotes)')
    axes[0].set_title(f'Reddit: r = {corr_reddit_pearson:.3f}, p = {p_corr_reddit_pearson:.4f}', fontweight='bold')
    axes[0].grid(alpha=0.3)

    # YouTube scatter
    axes[1].scatter(valid_youtube['vader_compound'], valid_youtube[engagement_col], alpha=0.3, s=20, color='#e74c3c')
    axes[1].set_xlabel('VADER Compound Score')
    axes[1].set_ylabel(f'YouTube {engagement_col}')
    axes[1].set_title(f'YouTube: r = {corr_youtube_pearson:.3f}, p = {p_corr_youtube_pearson:.4f}', fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
else:
    # Only Reddit plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid_reddit['vader_compound'], valid_reddit['score'], alpha=0.3, s=20, color='#3498db')
    ax.set_xlabel('VADER Compound Score')
    ax.set_ylabel('Reddit Score (Upvotes)')
    ax.set_title(f'Reddit: Sentiment × Engagement\nr = {corr_reddit_pearson:.3f}, p = {p_corr_reddit_pearson:.4f}', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
plt.savefig('statistical_tests_output/02_sentiment_engagement_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_sentiment_engagement_correlation.png")
print()

# ============================================================================
# 6. SUBJECTIVITY COMPARISON ACROSS STANCES
# ============================================================================
print("="*70)
print("TEST 5: ANOVA - SUBJECTIVITY ACROSS STANCES")
print("="*70)
print()

results_file.write("="*70 + "\n")
results_file.write("TEST 5: SUBJECTIVITY ANALYSIS (ANOVA)\n")
results_file.write("Hypothesis: Mean subjectivity is equal across stances\n")
results_file.write("-"*70 + "\n\n")

# Reddit subjectivity ANOVA
reddit_subj_groups = [reddit_df[reddit_df['Label'] == stance]['textblob_subjectivity'].dropna() 
                      for stance in ['P', 'I', 'N']]
f_stat_subj_reddit, p_subj_reddit = f_oneway(*reddit_subj_groups)

print("📊 Reddit: Subjectivity × Stance")
print(f"   F-statistic: {f_stat_subj_reddit:.4f}")
print(f"   P-value: {p_subj_reddit:.6f}")
print(f"   Result: {'SIGNIFICANT' if p_subj_reddit < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("REDDIT (Subjectivity by Stance):\n")
results_file.write(f"  F-statistic: {f_stat_subj_reddit:.4f}\n")
results_file.write(f"  P-value: {p_subj_reddit:.6f}\n")
for stance, name in [('P', 'Pro-Palestine'), ('I', 'Pro-Israel'), ('N', 'Neutral')]:
    mean_subj = reddit_df[reddit_df['Label'] == stance]['textblob_subjectivity'].mean()
    results_file.write(f"    {name}: {mean_subj:.4f}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Subjectivity DIFFERS across stances' if p_subj_reddit < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# YouTube subjectivity ANOVA
label_col_youtube = 'label' if 'label' in youtube_df.columns else 'Label'
youtube_subj_groups = [youtube_df[youtube_df[label_col_youtube] == stance]['textblob_subjectivity'].dropna() 
                       for stance in ['P', 'I', 'N']]
f_stat_subj_youtube, p_subj_youtube = f_oneway(*youtube_subj_groups)

print("📊 YouTube: Subjectivity × Stance")
print(f"   F-statistic: {f_stat_subj_youtube:.4f}")
print(f"   P-value: {p_subj_youtube:.6f}")
print(f"   Result: {'SIGNIFICANT' if p_subj_youtube < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")
print()

results_file.write("YOUTUBE (Subjectivity by Stance):\n")
results_file.write(f"  F-statistic: {f_stat_subj_youtube:.4f}\n")
results_file.write(f"  P-value: {p_subj_youtube:.6f}\n")
label_col_youtube = 'label' if 'label' in youtube_df.columns else 'Label'
for stance, name in [('P', 'Pro-Palestine'), ('I', 'Pro-Israel'), ('N', 'Neutral')]:
    mean_subj = youtube_df[youtube_df[label_col_youtube] == stance]['textblob_subjectivity'].mean()
    results_file.write(f"    {name}: {mean_subj:.4f}\n")
results_file.write(f"  Conclusion: {'REJECT null hypothesis - Subjectivity DIFFERS across stances' if p_subj_youtube < 0.05 else 'FAIL TO REJECT null hypothesis'}\n\n")

# ============================================================================
# 7. EFFECT SIZE CALCULATIONS (COHEN'S D)
# ============================================================================
print("="*70)
print("TEST 6: EFFECT SIZE - COHEN'S D")
print("="*70)
print()

results_file.write("="*70 + "\n")
results_file.write("TEST 6: EFFECT SIZE (COHEN'S D)\n")
results_file.write("Measures practical significance of platform differences\n")
results_file.write("-"*70 + "\n\n")

def cohens_d(group1, group2):
    """Calculate Cohen's d for effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

# Overall platform difference
d_overall = cohens_d(reddit_df['vader_compound'].dropna(), youtube_df['vader_compound'].dropna())
print(f"📊 Overall Platform Difference (Cohen's d): {d_overall:.4f}")
print(f"   Interpretation: {'Small' if abs(d_overall) < 0.5 else 'Medium' if abs(d_overall) < 0.8 else 'Large'} effect size")
print()

results_file.write("COHEN'S D (Effect Sizes):\n")
results_file.write(f"  Overall platform difference: {d_overall:.4f} ({'Small' if abs(d_overall) < 0.5 else 'Medium' if abs(d_overall) < 0.8 else 'Large'})\n")

# Pro-Palestine
d_p = cohens_d(reddit_p, youtube_p)
results_file.write(f"  Pro-Palestine stance: {d_p:.4f} ({'Small' if abs(d_p) < 0.5 else 'Medium' if abs(d_p) < 0.8 else 'Large'})\n")

# Pro-Israel
d_i = cohens_d(reddit_i, youtube_i)
results_file.write(f"  Pro-Israel stance: {d_i:.4f} ({'Small' if abs(d_i) < 0.5 else 'Medium' if abs(d_i) < 0.8 else 'Large'})\n\n")

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================
results_file.write("="*70 + "\n")
results_file.write("SUMMARY OF ALL STATISTICAL TESTS\n")
results_file.write("="*70 + "\n\n")

summary_data = {
    'Test': [
        'Chi-Square (Reddit)', 'Chi-Square (YouTube)',
        'ANOVA (Reddit)', 'ANOVA (YouTube)',
        'T-test (Overall)', 'T-test (Pro-P)', 'T-test (Pro-I)',
        'Correlation (Reddit)', 'Correlation (YouTube)',
        'ANOVA Subjectivity (Reddit)', 'ANOVA Subjectivity (YouTube)'
    ],
    'Statistic': [
        f'{chi2_reddit:.4f}', f'{chi2_youtube:.4f}',
        f'{f_stat_reddit:.4f}', f'{f_stat_youtube:.4f}',
        f'{t_stat_overall:.4f}', f'{t_stat_p:.4f}', f'{t_stat_i:.4f}',
        f'{corr_reddit_pearson:.4f}', f'{corr_youtube_pearson:.4f}',
        f'{f_stat_subj_reddit:.4f}', f'{f_stat_subj_youtube:.4f}'
    ],
    'P-value': [
        f'{p_reddit:.6f}', f'{p_youtube:.6f}',
        f'{p_anova_reddit:.6f}', f'{p_anova_youtube:.6f}',
        f'{p_ttest_overall:.6f}', f'{p_ttest_p:.6f}', f'{p_ttest_i:.6f}',
        f'{p_corr_reddit_pearson:.6f}', f'{p_corr_youtube_pearson:.6f}',
        f'{p_subj_reddit:.6f}', f'{p_subj_youtube:.6f}'
    ],
    'Significant': [
        '✓' if p_reddit < 0.05 else '✗',
        '✓' if p_youtube < 0.05 else '✗',
        '✓' if p_anova_reddit < 0.05 else '✗',
        '✓' if p_anova_youtube < 0.05 else '✗',
        '✓' if p_ttest_overall < 0.05 else '✗',
        '✓' if p_ttest_p < 0.05 else '✗',
        '✓' if p_ttest_i < 0.05 else '✗',
        '✓' if p_corr_reddit_pearson < 0.05 else '✗',
        '✓' if p_corr_youtube_pearson < 0.05 else '✗',
        '✓' if p_subj_reddit < 0.05 else '✗',
        '✓' if p_subj_youtube < 0.05 else '✗'
    ]
}

summary_df = pd.DataFrame(summary_data)
results_file.write(summary_df.to_string(index=False))
results_file.write("\n\n")

results_file.write("="*70 + "\n")
results_file.write("INTERPRETATION GUIDE\n")
results_file.write("="*70 + "\n")
results_file.write("✓ = Statistically significant (p < 0.05)\n")
results_file.write("✗ = Not statistically significant (p ≥ 0.05)\n\n")
results_file.write("Cohen's d interpretation:\n")
results_file.write("  |d| < 0.5 = Small effect\n")
results_file.write("  0.5 ≤ |d| < 0.8 = Medium effect\n")
results_file.write("  |d| ≥ 0.8 = Large effect\n")

results_file.close()
print("✓ Saved: statistical_test_results.txt")
print()

# ============================================================================
# CREATE SUMMARY VISUALIZATION
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Create a color-coded summary table
colors = ['#27ae60' if sig == '✓' else '#e74c3c' for sig in summary_df['Significant']]
y_pos = np.arange(len(summary_df))

ax.barh(y_pos, [1]*len(summary_df), color=colors, alpha=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels(summary_df['Test'])
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title('Statistical Tests Summary: Significance at α = 0.05', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Add significance markers
for i, (test, sig) in enumerate(zip(summary_df['Test'], summary_df['Significant'])):
    ax.text(0.5, i, sig, ha='center', va='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('statistical_tests_output/03_statistical_tests_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_statistical_tests_summary.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print("✅ STATISTICAL TESTING COMPLETE")
print("="*70)
print()
print("📁 Generated Files:")
print("  - statistical_test_results.txt (Comprehensive test results)")
print("  - 01_chi_square_contingency_tables.png (Stance × Sentiment)")
print("  - 02_sentiment_engagement_correlation.png (Engagement analysis)")
print("  - 03_statistical_tests_summary.png (All tests overview)")
print()
print("📊 Tests Conducted: 11")
print(f"   Significant results (p < 0.05): {summary_df['Significant'].value_counts().get('✓', 0)}")
print(f"   Non-significant results: {summary_df['Significant'].value_counts().get('✗', 0)}")
print()
