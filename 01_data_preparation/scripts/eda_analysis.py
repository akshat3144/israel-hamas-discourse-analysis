"""
Exploratory Data Analysis (EDA) for Israel-Hamas War Discourse
Analyzes Reddit and YouTube data to understand discourse patterns
Updated for new column schema:
  Reddit:  index, Label, Reasoning, self_text, comment_id, score, author_name,
           controversiality, created_time, subreddit, post_id, parent_id,
           permalink, post_title, post_score, post_upvote_ratio,
           post_created_time, num_comments
  YouTube: id, video id, author, text, likeCount, created_time, video_date,
           index, Label, Reasoning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import re

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / '01_data_preparation' / 'outputs'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - ISRAEL-HAMAS WAR DISCOURSE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df = pd.read_csv(DATA_DIR / 'reddit_labeled.csv')
youtube_df = pd.read_csv(DATA_DIR / 'youtube_labeled.csv')

# Standardise label column (both platforms use 'Label')
REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'
REDDIT_TEXT_COL   = 'self_text'
YOUTUBE_TEXT_COL  = 'text'

print(f"✔ Reddit data:  {len(reddit_df):,} rows")
print(f"✔ YouTube data: {len(youtube_df):,} rows")

# ============================================================================
# TYPE CONVERSION
# ============================================================================
print("\n🔧 Converting data types...")

reddit_num_cols = ['score', 'post_score', 'post_upvote_ratio',
                   'num_comments', 'controversiality']
for col in reddit_num_cols:
    if col in reddit_df.columns:
        reddit_df[col] = pd.to_numeric(reddit_df[col], errors='coerce')

# created_time is Unix timestamp (seconds)
for ts_col in ['created_time', 'post_created_time']:
    if ts_col in reddit_df.columns:
        reddit_df[ts_col] = pd.to_numeric(reddit_df[ts_col], errors='coerce')
        reddit_df[ts_col + '_dt'] = pd.to_datetime(
            reddit_df[ts_col], unit='s', errors='coerce', utc=True)

if 'likeCount' in youtube_df.columns:
    youtube_df['likeCount'] = pd.to_numeric(youtube_df['likeCount'], errors='coerce')
if 'created_time' in youtube_df.columns:
    youtube_df['created_time_dt'] = pd.to_datetime(
        youtube_df['created_time'], errors='coerce')
if 'video_date' in youtube_df.columns:
    youtube_df['video_date_dt'] = pd.to_datetime(
        youtube_df['video_date'], errors='coerce')

print("✔ Data type conversion complete")

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 80)

print("\n--- REDDIT DATA ---")
print(f"Total comments: {len(reddit_df):,}")
print(f"Columns: {list(reddit_df.columns)}")
print(f"\nData types:\n{reddit_df.dtypes}")

print("\n--- YOUTUBE DATA ---")
print(f"Total comments: {len(youtube_df):,}")
print(f"Columns: {list(youtube_df.columns)}")
print(f"\nData types:\n{youtube_df.dtypes}")

# ============================================================================
# 2. LABEL DISTRIBUTION
# ============================================================================
print("\n--- LABEL DISTRIBUTION ---")
print("\nReddit Stance Distribution:")
if REDDIT_LABEL_COL in reddit_df.columns:
    print(reddit_df[REDDIT_LABEL_COL].value_counts())
    print("\nPercentages:")
    print((reddit_df[REDDIT_LABEL_COL].value_counts(normalize=True) * 100).round(2))

print("\nYouTube Stance Distribution:")
if YOUTUBE_LABEL_COL in youtube_df.columns:
    print(youtube_df[YOUTUBE_LABEL_COL].value_counts())
    print("\nPercentages:")
    print((youtube_df[YOUTUBE_LABEL_COL].value_counts(normalize=True) * 100).round(2))

# ============================================================================
# 3. MISSING DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. MISSING DATA ANALYSIS")
print("=" * 80)

for name, df in [("REDDIT", reddit_df), ("YOUTUBE", youtube_df)]:
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({'Missing Count': miss, 'Percentage': miss_pct})
    print(f"\n--- {name} MISSING DATA ---")
    print(miss_df[miss_df['Missing Count'] > 0])

# ============================================================================
# 4. CONTENT LENGTH ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. CONTENT LENGTH ANALYSIS")
print("=" * 80)

reddit_df['text_length'] = (
    reddit_df[REDDIT_TEXT_COL].fillna('').astype(str).str.len())
youtube_df['text_length'] = (
    youtube_df[YOUTUBE_TEXT_COL].fillna('').astype(str).str.len())

reddit_df['word_count'] = (
    reddit_df[REDDIT_TEXT_COL].fillna('').astype(str)
    .apply(lambda x: len(x.split())))
youtube_df['word_count'] = (
    youtube_df[YOUTUBE_TEXT_COL].fillna('').astype(str)
    .apply(lambda x: len(x.split())))

print("\n--- REDDIT TEXT LENGTH (chars) ---")
print(reddit_df['text_length'].describe().round(2))
print(f"Median: {reddit_df['text_length'].median():.0f}")
print("\n--- YOUTUBE TEXT LENGTH (chars) ---")
print(youtube_df['text_length'].describe().round(2))
print(f"Median: {youtube_df['text_length'].median():.0f}")

# ============================================================================
# 5. ENGAGEMENT METRICS
# ============================================================================
print("\n" + "=" * 80)
print("4. ENGAGEMENT METRICS")
print("=" * 80)

if 'score' in reddit_df.columns:
    print(f"\nReddit Score Stats:\n{reddit_df['score'].describe().round(2)}")
    print(f"Median score: {reddit_df['score'].median()}")
if 'controversiality' in reddit_df.columns:
    print(f"\nControversiality Distribution:")
    print(reddit_df['controversiality'].value_counts())
if 'likeCount' in youtube_df.columns:
    print(f"\nYouTube Like Count Stats:\n{youtube_df['likeCount'].describe().round(2)}")
    print(f"Median likes: {youtube_df['likeCount'].median()}")

# ============================================================================
# 6. SUBREDDIT ANALYSIS (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("5. SUBREDDIT ANALYSIS (Reddit-specific)")
print("=" * 80)

if 'subreddit' in reddit_df.columns:
    subreddit_counts = reddit_df['subreddit'].value_counts()
    print(f"\nTotal unique subreddits: {len(subreddit_counts)}")
    print(f"\nTop 20 subreddits:\n{subreddit_counts.head(20)}")

    # Subreddit × Stance cross-tab
    sub_stance = pd.crosstab(
        reddit_df['subreddit'], reddit_df[REDDIT_LABEL_COL], normalize='index') * 100
    print(f"\nTop subreddits stance distribution (%):\n{sub_stance.head(15).round(1)}")

# ============================================================================
# 7. DATA QUALITY METRICS (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("6. DATA QUALITY METRICS")
print("=" * 80)

# Duplicate detection
reddit_dups = reddit_df.duplicated(subset=[REDDIT_TEXT_COL]).sum()
youtube_dups = youtube_df.duplicated(subset=[YOUTUBE_TEXT_COL]).sum()
print(f"\nReddit duplicate texts: {reddit_dups:,} ({reddit_dups/len(reddit_df)*100:.2f}%)")
print(f"YouTube duplicate texts: {youtube_dups:,} ({youtube_dups/len(youtube_df)*100:.2f}%)")

# Very short texts
reddit_short = (reddit_df['word_count'] < 3).sum()
youtube_short = (youtube_df['word_count'] < 3).sum()
print(f"\nReddit very short (<3 words): {reddit_short:,} ({reddit_short/len(reddit_df)*100:.2f}%)")
print(f"YouTube very short (<3 words): {youtube_short:,} ({youtube_short/len(youtube_df)*100:.2f}%)")

# ============================================================================
# 8. TEMPORAL ANALYSIS (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("7. TEMPORAL ANALYSIS")
print("=" * 80)

if 'created_time_dt' in reddit_df.columns:
    reddit_df['year_month'] = reddit_df['created_time_dt'].dt.to_period('M')
    temporal_reddit = reddit_df.groupby('year_month').size()
    print(f"\nReddit comment volume by month:\n{temporal_reddit}")

if 'created_time_dt' in youtube_df.columns:
    youtube_df['year_month'] = youtube_df['created_time_dt'].dt.to_period('M')
    temporal_youtube = youtube_df.groupby('year_month').size()
    print(f"\nYouTube comment volume by month:\n{temporal_youtube}")

# ============================================================================
# 9. REASONING ANALYSIS (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("8. ANNOTATION REASONING ANALYSIS")
print("=" * 80)

for name, df, label_col in [
        ("Reddit", reddit_df, REDDIT_LABEL_COL),
        ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
    if 'Reasoning' in df.columns:
        has_reasoning = df['Reasoning'].notna().sum()
        print(f"\n{name} - Comments with Reasoning notes: "
              f"{has_reasoning:,} ({has_reasoning/len(df)*100:.1f}%)")
        print(f"Avg Reasoning length: "
              f"{df['Reasoning'].fillna('').str.len().mean():.0f} chars")
        # Most common Reasoning snippets (first 50 chars)
        sample = df[df['Reasoning'].notna()]['Reasoning'].str[:60].value_counts().head(5)
        print(f"Sample Reasoning snippets:\n{sample}")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("9. GENERATING VISUALIZATIONS")
print("=" * 80)

colors = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}

# --- 01: Stance Distribution (Bar + Pie) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stance Distribution: Reddit vs YouTube', fontsize=16, fontweight='bold')

for ax, df, label_col, title, clr in [
        (axes[0, 0], reddit_df, REDDIT_LABEL_COL, 'Reddit – Count', 'Blues'),
        (axes[0, 1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube – Count', 'Reds')]:
    if label_col in df.columns:
        counts = df[label_col].value_counts()
        bar_colors = [colors.get(l, '#95a5a6') for l in counts.index]
        ax.bar(counts.index, counts.values, color=bar_colors, alpha=0.85, edgecolor='black')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Stance'); ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(counts.values):
            ax.text(i, v + max(counts.values) * 0.02, str(v),
                    ha='center', fontweight='bold')

for ax, df, label_col, title in [
        (axes[1, 0], reddit_df, REDDIT_LABEL_COL, 'Reddit – %'),
        (axes[1, 1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube – %')]:
    if label_col in df.columns:
        counts = df[label_col].value_counts()
        pie_colors = [colors.get(l, '#95a5a6') for l in counts.index]
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=pie_colors, explode=[0.05]*len(counts),
               shadow=True, startangle=90)
        ax.set_title(title, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_stance_distribution.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 01_stance_distribution.png")
plt.close()

# --- 02: Platform comparison bar ---
if REDDIT_LABEL_COL in reddit_df.columns and YOUTUBE_LABEL_COL in youtube_df.columns:
    r_pct = reddit_df[REDDIT_LABEL_COL].value_counts(normalize=True) * 100
    y_pct = youtube_df[YOUTUBE_LABEL_COL].value_counts(normalize=True) * 100
    cmp_df = pd.DataFrame({'Reddit': r_pct, 'YouTube': y_pct}).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmp_df.plot(kind='bar', ax=ax, width=0.75, alpha=0.85, edgecolor='black',
                color=['#3498db', '#e74c3c'])
    ax.set_title('Stance Distribution Comparison: Reddit vs YouTube (%)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Stance'); ax.set_ylabel('Percentage (%)')
    ax.legend(title='Platform'); ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_platform_stance_comparison.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 02_platform_stance_comparison.png")
    plt.close()

# --- 03: Text length distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(reddit_df['text_length'].clip(upper=2000), bins=60,
             color='#3498db', alpha=0.75, edgecolor='black')
axes[0].axvline(reddit_df['text_length'].median(), color='red',
                linestyle='--', label=f'Median: {reddit_df["text_length"].median():.0f}')
axes[0].set_title('Reddit – Text Length Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Characters (clipped at 2000)')
axes[0].set_ylabel('Frequency'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(youtube_df['text_length'].clip(upper=1000), bins=60,
             color='#e74c3c', alpha=0.75, edgecolor='black')
axes[1].axvline(youtube_df['text_length'].median(), color='navy',
                linestyle='--', label=f'Median: {youtube_df["text_length"].median():.0f}')
axes[1].set_title('YouTube – Text Length Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Characters (clipped at 1000)')
axes[1].set_ylabel('Frequency'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_text_length_distribution.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 03_text_length_distribution.png")
plt.close()

# --- 04: Text length by stance (box plots) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
palette = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}

if REDDIT_LABEL_COL in reddit_df.columns:
    plot_df = reddit_df[reddit_df[REDDIT_LABEL_COL].isin(['P', 'I', 'N'])].copy()
    plot_df['text_length_clip'] = plot_df['text_length'].clip(upper=2000)
    sns.boxplot(data=plot_df, x=REDDIT_LABEL_COL, y='text_length_clip',
                palette=palette, ax=axes[0], showfliers=False,
                order=['P', 'I', 'N'])
    axes[0].set_title('Reddit – Comment Length by Stance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Stance'); axes[0].set_ylabel('Text Length (chars)')

if YOUTUBE_LABEL_COL in youtube_df.columns:
    plot_df_y = youtube_df[youtube_df[YOUTUBE_LABEL_COL].isin(['P', 'I', 'N'])].copy()
    plot_df_y['text_length_clip'] = plot_df_y['text_length'].clip(upper=1000)
    sns.boxplot(data=plot_df_y, x=YOUTUBE_LABEL_COL, y='text_length_clip',
                palette=palette, ax=axes[1], showfliers=False,
                order=['P', 'I', 'N'])
    axes[1].set_title('YouTube – Comment Length by Stance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Stance'); axes[1].set_ylabel('Text Length (chars)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_text_length_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 04_text_length_by_stance.png")
plt.close()

# --- 05: Reddit engagement by stance ---
if 'score' in reddit_df.columns and REDDIT_LABEL_COL in reddit_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    valid = reddit_df[reddit_df[REDDIT_LABEL_COL].isin(['P', 'I', 'N'])].copy()
    valid['score_clip'] = valid['score'].clip(lower=-100, upper=500)

    sns.boxplot(data=valid, x=REDDIT_LABEL_COL, y='score_clip',
                palette=palette, ax=axes[0], showfliers=False, order=['P', 'I', 'N'])
    axes[0].set_title('Reddit – Score by Stance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Stance'); axes[0].set_ylabel('Score (upvotes)')

    stance_means = valid.groupby(REDDIT_LABEL_COL)['score'].mean().reindex(['P', 'I', 'N'])
    bar_colors = [colors.get(l, '#95a5a6') for l in stance_means.index]
    axes[1].bar(stance_means.index, stance_means.values,
                color=bar_colors, alpha=0.85, edgecolor='black')
    axes[1].set_title('Reddit – Mean Score by Stance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Stance'); axes[1].set_ylabel('Mean Score')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_reddit_engagement_by_stance.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 05_reddit_engagement_by_stance.png")
    plt.close()

# --- 06: Subreddit analysis (NEW) ---
if 'subreddit' in reddit_df.columns:
    top_subs = reddit_df['subreddit'].value_counts().head(20)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Top subreddits by comment count
    axes[0].barh(top_subs.index[::-1], top_subs.values[::-1],
                 color='#3498db', alpha=0.85, edgecolor='black')
    axes[0].set_title('Top 20 Subreddits by Comment Count',
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Comments')
    axes[0].grid(axis='x', alpha=0.3)

    # Subreddit stance composition (stacked bar for top 15)
    top15_subs = top_subs.head(15).index
    sub_stance_abs = pd.crosstab(
        reddit_df[reddit_df['subreddit'].isin(top15_subs)]['subreddit'],
        reddit_df[reddit_df['subreddit'].isin(top15_subs)][REDDIT_LABEL_COL]
    ).reindex(top15_subs)
    sub_stance_pct = sub_stance_abs.div(sub_stance_abs.sum(axis=1), axis=0) * 100

    stance_plot_cols = [c for c in ['P', 'I', 'N'] if c in sub_stance_pct.columns]
    sub_stance_pct[stance_plot_cols].plot(
        kind='barh', stacked=True, ax=axes[1],
        color=[colors.get(c, '#95a5a6') for c in stance_plot_cols],
        alpha=0.85, edgecolor='black')
    axes[1].set_title('Stance Composition per Subreddit (Top 15)',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Percentage (%)')
    axes[1].legend(title='Stance', labels=['Pro-Palestine', 'Pro-Israel', 'Neutral'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_subreddit_analysis.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 06_subreddit_analysis.png")
    plt.close()

# --- 07: Temporal volume analysis (NEW) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Monthly Comment Volume by Stance', fontsize=15, fontweight='bold')

if 'year_month' in reddit_df.columns and REDDIT_LABEL_COL in reddit_df.columns:
    r_temporal = reddit_df.groupby(
        ['year_month', REDDIT_LABEL_COL]).size().unstack(fill_value=0)
    r_temporal.index = r_temporal.index.astype(str)
    stance_cols = [c for c in ['P', 'I', 'N'] if c in r_temporal.columns]
    for s in stance_cols:
        axes[0].plot(r_temporal.index, r_temporal[s],
                     marker='o', label=f'{s}', color=colors.get(s),
                     linewidth=2, markersize=5)
    axes[0].set_title('Reddit – Monthly Comment Volume', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Month'); axes[0].set_ylabel('Comment Count')
    axes[0].legend(title='Stance'); axes[0].grid(alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

if 'year_month' in youtube_df.columns and YOUTUBE_LABEL_COL in youtube_df.columns:
    y_temporal = youtube_df.groupby(
        ['year_month', YOUTUBE_LABEL_COL]).size().unstack(fill_value=0)
    y_temporal.index = y_temporal.index.astype(str)
    stance_cols_y = [c for c in ['P', 'I', 'N'] if c in y_temporal.columns]
    for s in stance_cols_y:
        axes[1].plot(y_temporal.index, y_temporal[s],
                     marker='o', label=f'{s}', color=colors.get(s),
                     linewidth=2, markersize=5)
    axes[1].set_title('YouTube – Monthly Comment Volume', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Month'); axes[1].set_ylabel('Comment Count')
    axes[1].legend(title='Stance'); axes[1].grid(alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_temporal_volume.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 07_temporal_volume.png")
plt.close()

# --- 08: Controversiality by stance (Reddit) ---
if 'controversiality' in reddit_df.columns and REDDIT_LABEL_COL in reddit_df.columns:
    controversy_cross = pd.crosstab(
        reddit_df[REDDIT_LABEL_COL], reddit_df['controversiality'],
        normalize='index') * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    controversy_cross.plot(kind='bar', ax=ax, stacked=False, alpha=0.85, edgecolor='black')
    ax.set_title('Reddit Controversiality by Stance (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stance'); ax.set_ylabel('Percentage (%)')
    ax.legend(title='Controversial', labels=['No', 'Yes'])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_controversiality_by_stance.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 08_controversiality_by_stance.png")
    plt.close()

# --- 09: YouTube engagement (likes) by stance ---
if 'likeCount' in youtube_df.columns and YOUTUBE_LABEL_COL in youtube_df.columns:
    valid_yt = youtube_df[youtube_df[YOUTUBE_LABEL_COL].isin(['P', 'I', 'N'])].copy()
    valid_yt['likeCount_clip'] = valid_yt['likeCount'].clip(upper=500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=valid_yt, x=YOUTUBE_LABEL_COL, y='likeCount_clip',
                palette=palette, ax=axes[0], showfliers=False, order=['P', 'I', 'N'])
    axes[0].set_title('YouTube – Likes by Stance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Stance'); axes[0].set_ylabel('Like Count')

    like_means = valid_yt.groupby(YOUTUBE_LABEL_COL)['likeCount'].mean().reindex(['P', 'I', 'N'])
    bar_c = [colors.get(l, '#95a5a6') for l in like_means.index]
    axes[1].bar(like_means.index, like_means.values,
                color=bar_c, alpha=0.85, edgecolor='black')
    axes[1].set_title('YouTube – Mean Likes by Stance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Stance'); axes[1].set_ylabel('Mean Likes')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_youtube_engagement_by_stance.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 09_youtube_engagement_by_stance.png")
    plt.close()

# --- 10: User activity distribution (NEW) ---
if 'author_name' in reddit_df.columns:
    user_counts = reddit_df['author_name'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(user_counts.values, bins=50, color='#3498db',
                 alpha=0.8, edgecolor='black', log=True)
    axes[0].set_title('Reddit – User Activity Distribution (log scale)',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Comments per User'); axes[0].set_ylabel('Number of Users (log)')
    axes[0].grid(alpha=0.3)

    # Cumulative % of comments by top users
    sorted_counts = np.sort(user_counts.values)[::-1]
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum() * 100
    top_pct = np.arange(1, len(sorted_counts)+1) / len(sorted_counts) * 100
    axes[1].plot(top_pct, cumulative, color='#e74c3c', linewidth=2)
    axes[1].set_title('Reddit – User Participation Inequality (Lorenz Curve)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Top X% of Users'); axes[1].set_ylabel('% of Total Comments')
    axes[1].axvline(10, color='gray', linestyle='--', alpha=0.5, label='Top 10%')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_user_activity_distribution.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 10_user_activity_distribution.png")
    plt.close()

if 'author' in youtube_df.columns:
    yt_user_counts = youtube_df['author'].value_counts()
    top10_users_yt = yt_user_counts.head(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top10_users_yt.index[::-1], top10_users_yt.values[::-1],
            color='#e74c3c', alpha=0.85, edgecolor='black')
    ax.set_title('YouTube – Top 20 Most Active Commenters',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Comments'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_youtube_top_users.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 11_youtube_top_users.png")
    plt.close()

# ============================================================================
# 11. EXPORT CLEANED DATA
# ============================================================================
print("\n" + "=" * 80)
print("10. EXPORTING PROCESSED DATA")
print("=" * 80)

# Drop internal helper columns before export
reddit_export = reddit_df.copy()
youtube_export = youtube_df.copy()

reddit_export.to_csv(OUTPUT_DIR / 'reddit_processed.csv', index=False, encoding='utf-8')
youtube_export.to_csv(OUTPUT_DIR / 'youtube_processed.csv', index=False, encoding='utf-8')
print("✔ Saved: reddit_processed.csv")
print("✔ Saved: youtube_processed.csv")

# ============================================================================
# 12. SUMMARY REPORT
# ============================================================================
with open(OUTPUT_DIR / 'eda_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS SUMMARY REPORT\n")
    f.write("Israel-Hamas War Discourse Analysis\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. DATASET OVERVIEW\n" + "-" * 80 + "\n")
    f.write(f"Reddit Comments:  {len(reddit_df):,}\n")
    f.write(f"YouTube Comments: {len(youtube_df):,}\n")
    f.write(f"Total:            {len(reddit_df) + len(youtube_df):,}\n\n")

    f.write("2. STANCE DISTRIBUTION\n" + "-" * 80 + "\n")
    for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                             ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
        if lcol in df.columns:
            f.write(f"\n{name}:\n{df[lcol].value_counts()}\n")
            f.write("Percentages:\n")
            f.write(str((df[lcol].value_counts(normalize=True)*100).round(2)) + "\n")

    f.write("\n3. TEXT LENGTH STATISTICS\n" + "-" * 80 + "\n")
    f.write(f"\nReddit (chars):\n{reddit_df['text_length'].describe().round(2)}\n")
    f.write(f"\nYouTube (chars):\n{youtube_df['text_length'].describe().round(2)}\n")

    f.write("\n4. ENGAGEMENT STATISTICS\n" + "-" * 80 + "\n")
    if 'score' in reddit_df.columns:
        f.write(f"\nReddit Score:\n{reddit_df['score'].describe().round(2)}\n")
    if 'likeCount' in youtube_df.columns:
        f.write(f"\nYouTube Likes:\n{youtube_df['likeCount'].describe().round(2)}\n")

    if 'subreddit' in reddit_df.columns:
        f.write("\n5. TOP SUBREDDITS\n" + "-" * 80 + "\n")
        f.write(str(reddit_df['subreddit'].value_counts().head(20)) + "\n")

    f.write("\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n")

print("✔ Saved: eda_summary_report.txt")

print("\n" + "=" * 80)
print("✅ EDA COMPLETE!")
print("=" * 80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for i, name in enumerate([
    "01_stance_distribution.png",
    "02_platform_stance_comparison.png",
    "03_text_length_distribution.png",
    "04_text_length_by_stance.png",
    "05_reddit_engagement_by_stance.png",
    "06_subreddit_analysis.png",
    "07_temporal_volume.png",
    "08_controversiality_by_stance.png",
    "09_youtube_engagement_by_stance.png",
    "10_user_activity_distribution.png",
    "11_youtube_top_users.png",
    "eda_summary_report.txt",
    "reddit_processed.csv",
    "youtube_processed.csv",
], start=1):
    print(f"  {i:02d}. {name}")
print("=" * 80)
