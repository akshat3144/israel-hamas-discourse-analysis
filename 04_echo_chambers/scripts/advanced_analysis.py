"""
Advanced Analysis — RQ3
Regression models, algorithmic amplification (controversiality),
and narrative complexity (readability) analysis.
Column schema: Reddit uses self_text + Label + score + controversiality;
               YouTube uses text + Label + likeCount.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import textstat
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '04_echo_chambers' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_TEXT_COL   = 'self_text'
YOUTUBE_TEXT_COL  = 'text'
REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

print("=" * 80)
print("ADVANCED ANALYSIS — REGRESSION, AMPLIFICATION, COMPLEXITY")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df  = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')
youtube_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'youtube_with_sentiment.csv')

print(f"✔ Reddit:  {len(reddit_df):,} rows")
print(f"✔ YouTube: {len(youtube_df):,} rows")

# ============================================================================
# 1. REGRESSION: Score ~ Sentiment + Stance (Reddit)
# ============================================================================
print("\n" + "=" * 80)
print("1. REGRESSION: SCORE ~ SENTIMENT + STANCE (Reddit)")
print("=" * 80)

reddit_df['score'] = pd.to_numeric(reddit_df['score'], errors='coerce')
reg_df = reddit_df[['score', 'vader_compound', REDDIT_LABEL_COL,
                     'textblob_subjectivity', 'text_length']].dropna()
reg_df = reg_df[reg_df[REDDIT_LABEL_COL].isin(['P', 'I', 'N'])].copy()

print(f"Data points for regression: {len(reg_df):,}")

if len(reg_df) > 10:
    formula = f'score ~ vader_compound + textblob_subjectivity + text_length + C({REDDIT_LABEL_COL})'
    model = ols(formula, data=reg_df).fit()
    print("\n--- Regression Results ---")
    print(model.summary())

    with open(OUTPUT_DIR / 'regression_results.txt', 'w', encoding='utf-8') as f:
        f.write(str(model.summary()))
    print("✔ Saved: regression_results.txt")

    # Coefficient plot
    params = model.params.drop('Intercept')
    conf   = model.conf_int().drop('Intercept')
    conf.columns = ['Lower', 'Upper']
    err = params - conf['Lower']

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if v < 0 else '#27ae60' for v in params.values]
    ax.barh(params.index, params.values,
            xerr=err.values, color=colors, alpha=0.8, edgecolor='black', capsize=4)
    ax.axvline(0, color='black', linestyle='--', alpha=0.6)
    ax.set_title('Regression Coefficients: Impact on Reddit Score',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Coefficient Value')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_regression_coefficients.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 01_regression_coefficients.png")
    plt.close()
else:
    print("⚠️  Not enough data for regression.")

# ============================================================================
# 2. ALGORITHMIC AMPLIFICATION (Reddit controversiality)
# ============================================================================
print("\n" + "=" * 80)
print("2. ALGORITHMIC AMPLIFICATION (Controversiality)")
print("=" * 80)

if 'controversiality' in reddit_df.columns:
    amp_df = reddit_df[['score', 'controversiality', REDDIT_LABEL_COL]].dropna()
    amp_df = amp_df[amp_df[REDDIT_LABEL_COL].isin(['P', 'I', 'N'])].copy()
    amp_df['controversiality'] = amp_df['controversiality'].astype(int)

    mean_scores = amp_df.groupby('controversiality')['score'].agg(['mean', 'median', 'count'])
    print(f"\nMean Score by Controversiality:\n{mean_scores.round(2)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Algorithmic Amplification via Controversiality', fontsize=14, fontweight='bold')

    sns.barplot(x='controversiality', y='score', data=amp_df, ax=axes[0], palette='viridis',
                order=[0, 1], errorbar='sd')
    axes[0].set_title('Mean Score by Controversiality', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Controversial (0=No, 1=Yes)')
    axes[0].set_ylabel('Average Score')
    axes[0].set_xticklabels(['Non-Controversial', 'Controversial'])
    axes[0].grid(axis='y', alpha=0.3)

    sns.barplot(x=REDDIT_LABEL_COL, y='score', hue='controversiality',
                data=amp_df, ax=axes[1], palette='viridis',
                order=['P', 'I', 'N'], errorbar='sd')
    axes[1].set_title('Controversiality Amplification by Stance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Stance'); axes[1].set_ylabel('Average Score')
    axes[1].legend(title='Controversial', labels=['No', 'Yes'])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '12_amplification_controversy.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 02_amplification_controversy.png")
    plt.close()

    # Controversy × Stance heatmap
    ct = pd.crosstab(amp_df[REDDIT_LABEL_COL], amp_df['controversiality'], normalize='index') * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                xticklabels=['Non-Controversial', 'Controversial'])
    ax.set_title('Controversiality Rate by Stance (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('Stance')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '13_amplification_by_stance.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 03_amplification_by_stance.png")
    plt.close()
else:
    print("⚠️  Controversiality column not found.")

# ============================================================================
# 3. NARRATIVE COMPLEXITY — READABILITY (Flesch Reading Ease)
# ============================================================================
print("\n" + "=" * 80)
print("3. NARRATIVE COMPLEXITY (Readability)")
print("=" * 80)

def calculate_readability(text):
    if pd.isna(text) or str(text).strip() == '':
        return np.nan
    try:
        return textstat.flesch_reading_ease(str(text))
    except Exception:
        return np.nan

def calculate_word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

print("Calculating readability scores for Reddit...")
reddit_df['readability'] = reddit_df[REDDIT_TEXT_COL].apply(calculate_readability)
print("Calculating readability scores for YouTube...")
youtube_df['readability'] = youtube_df[YOUTUBE_TEXT_COL].apply(calculate_readability)

# Filter extreme outliers (textstat can return very negative for non-English)
reddit_df  = reddit_df[reddit_df['readability'].between(-100, 150)].copy()
youtube_df = youtube_df[youtube_df['readability'].between(-100, 150)].copy()

print(f"\nReddit  Mean Readability: {reddit_df['readability'].mean():.2f}")
print(f"YouTube Mean Readability: {youtube_df['readability'].mean():.2f}")
print("(Higher = easier to read | ~60-70 = standard | <30 = college level)")

from scipy.stats import ttest_ind, mannwhitneyu
r_read = reddit_df['readability'].dropna()
y_read = youtube_df['readability'].dropna()
t, p = ttest_ind(r_read, y_read)
u, p_mw = mannwhitneyu(r_read, y_read, alternative='two-sided')
print(f"\nPlatform Readability Difference: t={t:.4f}, p={p:.6f}")
print(f"Mann-Whitney U: U={u:.0f}, p={p_mw:.6f}")

# ── 04: Platform comparison box ──
fig, ax = plt.subplots(figsize=(9, 6))
data_to_plot = [r_read, y_read]
bplot = ax.boxplot(data_to_plot, labels=['Reddit', 'YouTube'], patch_artist=True,
                   notch=True, showfliers=False)
bplot['boxes'][0].set_facecolor('#3498db')
bplot['boxes'][1].set_facecolor('#e74c3c')
ax.set_title('Narrative Complexity: Readability Scores by Platform',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Flesch Reading Ease Score')
ax.text(0.98, 0.98, f't={t:.3f}, p={p:.4f}',
        ha='right', va='top', transform=ax.transAxes, fontsize=10, color='gray')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '14_complexity_platform_comparison.png', dpi=300, bbox_inches='tight')
print("\n✔ Saved: 04_complexity_platform_comparison.png")
plt.close()

# ── 05: Readability by stance ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df, lcol, title, color in [
        (axes[0], reddit_df,  REDDIT_LABEL_COL,  'Reddit',  '#3498db'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube', '#e74c3c')]:
    if lcol in df.columns:
        valid = df[df[lcol].isin(['P', 'I', 'N'])]
        sns.boxplot(x=lcol, y='readability', data=valid, palette={'P':'#2ecc71','I':'#3498db','N':'#95a5a6'},
                    ax=ax, showfliers=False, order=['P', 'I', 'N'])
        ax.set_title(f'{title} Complexity by Stance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Flesch Reading Ease'); ax.set_xlabel('Stance')
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '15_complexity_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 05_complexity_by_stance.png")
plt.close()

# ── 06: Subreddit-level readability (NEW) ──
if 'subreddit' in reddit_df.columns:
    top_subs = reddit_df['subreddit'].value_counts().head(12).index
    sub_read = (reddit_df[reddit_df['subreddit'].isin(top_subs)]
                .groupby('subreddit')['readability']
                .agg(['mean', 'median', 'count'])
                .sort_values('mean'))
    print(f"\nReadability by Subreddit:\n{sub_read.round(2)}")

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = ['#e74c3c' if v < 50 else '#27ae60' for v in sub_read['mean']]
    ax.barh(sub_read.index, sub_read['mean'], color=bar_colors, alpha=0.85, edgecolor='black')
    ax.axvline(50, color='gray', linestyle='--', alpha=0.6, label='Score=50 (standard)')
    ax.set_title('Mean Readability by Subreddit (Reddit)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Flesch Reading Ease Score'); ax.legend(); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '16_readability_by_subreddit.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 06_readability_by_subreddit.png")
    plt.close()

print("\n" + "=" * 80)
print("✅ ADVANCED ANALYSIS COMPLETE")
print("=" * 80)
