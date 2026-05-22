"""
Sentiment Analysis for Israel-Hamas War Discourse
RQ1: How does emotional tone differ between platforms?
Column schema:
  Reddit:  self_text, Label, author_name, score, subreddit, created_time (Unix)
  YouTube: text, Label, likeCount, created_time, video_date
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EDA_OUTPUT_DIR = ROOT_DIR / '01_data_preparation' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column constants
REDDIT_TEXT_COL   = 'self_text'
YOUTUBE_TEXT_COL  = 'text'
REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

print("=" * 80)
print("SENTIMENT ANALYSIS - ISRAEL-HAMAS WAR DISCOURSE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading processed data...")
reddit_df  = pd.read_csv(EDA_OUTPUT_DIR / 'reddit_processed.csv')
youtube_df = pd.read_csv(EDA_OUTPUT_DIR / 'youtube_processed.csv')

# Restore datetime columns (Unix timestamp for Reddit)
if 'created_time' in reddit_df.columns:
    reddit_df['created_time'] = pd.to_numeric(reddit_df['created_time'], errors='coerce')
    reddit_df['created_time_dt'] = pd.to_datetime(
        reddit_df['created_time'], unit='s', errors='coerce', utc=True)
    reddit_df['year_month'] = reddit_df['created_time_dt'].dt.to_period('M')

if 'created_time' in youtube_df.columns:
    youtube_df['created_time_dt'] = pd.to_datetime(
        youtube_df['created_time'], errors='coerce')
    youtube_df['year_month'] = youtube_df['created_time_dt'].dt.to_period('M')

print(f"✔ Reddit: {len(reddit_df):,} rows")
print(f"✔ YouTube: {len(youtube_df):,} rows")

# ============================================================================
# SENTIMENT ANALYZERS
# ============================================================================
print("\n🔧 Initializing sentiment analyzers...")
vader = SentimentIntensityAnalyzer()
print("✔ VADER Sentiment Analyzer loaded")
print("✔ TextBlob Sentiment Analyzer loaded")

# ============================================================================
# SENTIMENT FUNCTIONS
# ============================================================================

def get_vader_sentiment(text):
    """Return VADER scores + label for a single text."""
    if pd.isna(text) or str(text).strip() == '':
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0, 'label': 'neutral'}
    scores = vader.polarity_scores(str(text))
    if scores['compound'] >= 0.05:
        scores['label'] = 'positive'
    elif scores['compound'] <= -0.05:
        scores['label'] = 'negative'
    else:
        scores['label'] = 'neutral'
    return scores


def get_textblob_sentiment(text):
    """Return TextBlob polarity, subjectivity + label."""
    if pd.isna(text) or str(text).strip() == '':
        return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'neutral'}
    blob = TextBlob(str(text))
    polarity    = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.1:
        label = 'positive'
    elif polarity < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    return {'polarity': polarity, 'subjectivity': subjectivity, 'label': label}

# ============================================================================
# ANALYZE REDDIT
# ============================================================================
print("\n" + "=" * 80)
print("ANALYZING REDDIT SENTIMENT")
print("=" * 80)

print(f"\n🔍 Running VADER on Reddit [{REDDIT_TEXT_COL}]...")
reddit_vader = reddit_df[REDDIT_TEXT_COL].fillna('').astype(str).apply(get_vader_sentiment)
reddit_df['vader_compound'] = reddit_vader.apply(lambda x: x['compound'])
reddit_df['vader_pos']      = reddit_vader.apply(lambda x: x['pos'])
reddit_df['vader_neu']      = reddit_vader.apply(lambda x: x['neu'])
reddit_df['vader_neg']      = reddit_vader.apply(lambda x: x['neg'])
reddit_df['vader_label']    = reddit_vader.apply(lambda x: x['label'])
print("✔ VADER complete")

print("\n🔍 Running TextBlob on Reddit...")
reddit_tb = reddit_df[REDDIT_TEXT_COL].fillna('').astype(str).apply(get_textblob_sentiment)
reddit_df['textblob_polarity']     = reddit_tb.apply(lambda x: x['polarity'])
reddit_df['textblob_subjectivity'] = reddit_tb.apply(lambda x: x['subjectivity'])
reddit_df['textblob_label']        = reddit_tb.apply(lambda x: x['label'])
print("✔ TextBlob complete")

print("\n📊 REDDIT SENTIMENT DISTRIBUTION (VADER):")
print(reddit_df['vader_label'].value_counts())
print("\nPercentages:")
print((reddit_df['vader_label'].value_counts(normalize=True) * 100).round(2))

# ============================================================================
# ANALYZE YOUTUBE
# ============================================================================
print("\n" + "=" * 80)
print("ANALYZING YOUTUBE SENTIMENT")
print("=" * 80)

print(f"\n🔍 Running VADER on YouTube [{YOUTUBE_TEXT_COL}]...")
youtube_vader = youtube_df[YOUTUBE_TEXT_COL].fillna('').astype(str).apply(get_vader_sentiment)
youtube_df['vader_compound'] = youtube_vader.apply(lambda x: x['compound'])
youtube_df['vader_pos']      = youtube_vader.apply(lambda x: x['pos'])
youtube_df['vader_neu']      = youtube_vader.apply(lambda x: x['neu'])
youtube_df['vader_neg']      = youtube_vader.apply(lambda x: x['neg'])
youtube_df['vader_label']    = youtube_vader.apply(lambda x: x['label'])
print("✔ VADER complete")

print("\n🔍 Running TextBlob on YouTube...")
youtube_tb = youtube_df[YOUTUBE_TEXT_COL].fillna('').astype(str).apply(get_textblob_sentiment)
youtube_df['textblob_polarity']     = youtube_tb.apply(lambda x: x['polarity'])
youtube_df['textblob_subjectivity'] = youtube_tb.apply(lambda x: x['subjectivity'])
youtube_df['textblob_label']        = youtube_tb.apply(lambda x: x['label'])
print("✔ TextBlob complete")

print("\n📊 YOUTUBE SENTIMENT DISTRIBUTION (VADER):")
print(youtube_df['vader_label'].value_counts())
print("\nPercentages:")
print((youtube_df['vader_label'].value_counts(normalize=True) * 100).round(2))

# ============================================================================
# SENTIMENT BY STANCE
# ============================================================================
print("\n" + "=" * 80)
print("SENTIMENT BY STANCE ANALYSIS")
print("=" * 80)

for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                        ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
    print(f"\n📊 {name} – Mean VADER by Stance:")
    grp = df.groupby(lcol).agg(
        compound_mean=('vader_compound', 'mean'),
        compound_median=('vader_compound', 'median'),
        compound_std=('vader_compound', 'std'),
        pos_mean=('vader_pos', 'mean'),
        neg_mean=('vader_neg', 'mean'),
        neu_mean=('vader_neu', 'mean'),
        subjectivity_mean=('textblob_subjectivity', 'mean'),
        polarity_mean=('textblob_polarity', 'mean'),
    ).round(4)
    print(grp)

# ============================================================================
# SENTIMENT BY SUBREDDIT (NEW)
# ============================================================================
if 'subreddit' in reddit_df.columns:
    print("\n" + "=" * 80)
    print("SENTIMENT BY SUBREDDIT (Reddit)")
    print("=" * 80)
    sub_sentiment = (reddit_df.groupby('subreddit')['vader_compound']
                     .agg(['mean', 'median', 'count'])
                     .rename(columns={'mean': 'mean_compound',
                                      'median': 'median_compound',
                                      'count': 'n_comments'})
                     .sort_values('n_comments', ascending=False)
                     .head(20)
                     .round(4))
    print(sub_sentiment)

# ============================================================================
# TEMPORAL SENTIMENT TRENDS (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("TEMPORAL SENTIMENT TRENDS (NEW)")
print("=" * 80)

for name, df in [("Reddit", reddit_df), ("YouTube", youtube_df)]:
    if 'year_month' in df.columns:
        t_sent = (df.groupby('year_month')['vader_compound']
                  .agg(['mean', 'count'])
                  .rename(columns={'mean': 'mean_compound', 'count': 'n'}))
        t_sent.index = t_sent.index.astype(str)
        print(f"\n{name} – Monthly Mean Sentiment:\n{t_sent.round(4)}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

palette = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}
sent_colors = {'positive': '#27ae60', 'neutral': '#95a5a6', 'negative': '#e74c3c'}

# --- 01: Sentiment distribution 2×2 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sentiment Distribution by Platform & Method', fontsize=15, fontweight='bold')

for ax, df, col, title, c in [
        (axes[0, 0], reddit_df,  'vader_label',    'Reddit – VADER',    '#3498db'),
        (axes[0, 1], youtube_df, 'vader_label',    'YouTube – VADER',   '#e74c3c'),
        (axes[1, 0], reddit_df,  'textblob_label', 'Reddit – TextBlob', '#3498db'),
        (axes[1, 1], youtube_df, 'textblob_label', 'YouTube – TextBlob','#e74c3c')]:
    counts = df[col].value_counts()
    bar_c  = [sent_colors.get(s, '#95a5a6') for s in counts.index]
    ax.bar(counts.index, counts.values, color=bar_c, alpha=0.85, edgecolor='black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentiment'); ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values)*0.02,
                f'{v}\n({v/counts.sum()*100:.1f}%)', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 01_sentiment_distribution.png")
plt.close()

# --- 02: Sentiment × Stance heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df, lcol, title, cmap in [
        (axes[0], reddit_df,  REDDIT_LABEL_COL,  'Reddit: Sentiment × Stance (%)',  'RdYlGn'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube: Sentiment × Stance (%)', 'RdYlGn')]:
    if lcol in df.columns:
        ct = pd.crosstab(df[lcol], df['vader_label'], normalize='index') * 100
        # Reorder rows/cols for consistency
        ct = ct.reindex(index=[c for c in ['P', 'I', 'N'] if c in ct.index],
                        columns=[c for c in ['positive', 'neutral', 'negative'] if c in ct.columns])
        sns.heatmap(ct, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                    cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=70)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentiment (VADER)'); ax.set_ylabel('Stance')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_sentiment_by_stance_heatmap.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 02_sentiment_by_stance_heatmap.png")
plt.close()

# --- 03: VADER compound score by stance (box) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df, lcol, title in [
        (axes[0], reddit_df,  REDDIT_LABEL_COL,  'Reddit: VADER Compound × Stance'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube: VADER Compound × Stance')]:
    if lcol in df.columns:
        valid = df[df[lcol].isin(['P', 'I', 'N'])]
        sns.boxplot(data=valid, x=lcol, y='vader_compound', palette=palette,
                    ax=ax, showfliers=False, order=['P', 'I', 'N'])
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Stance'); ax.set_ylabel('VADER Compound Score')
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_compound_score_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 03_compound_score_by_stance.png")
plt.close()

# --- 04: Polarity vs Subjectivity scatter ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df, lcol, title in [
        (axes[0], reddit_df,  REDDIT_LABEL_COL,  'Reddit: Polarity vs Subjectivity'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube: Polarity vs Subjectivity')]:
    if lcol in df.columns:
        for stance in ['P', 'I', 'N']:
            sub = df[df[lcol] == stance]
            ax.scatter(sub['textblob_subjectivity'], sub['textblob_polarity'],
                       alpha=0.25, s=15, c=palette.get(stance, '#95a5a6'), label=stance)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Subjectivity'); ax.set_ylabel('Polarity')
        ax.legend(title='Stance'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_polarity_subjectivity_scatter.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 04_polarity_subjectivity_scatter.png")
plt.close()

# --- 05: Platform comparison bar ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, method, col, title in [
        (axes[0], 'VADER',    'vader_label',    'Sentiment Comparison – VADER'),
        (axes[1], 'TextBlob', 'textblob_label', 'Sentiment Comparison – TextBlob')]:
    cmp = pd.DataFrame({
        'Reddit':  reddit_df[col].value_counts(normalize=True) * 100,
        'YouTube': youtube_df[col].value_counts(normalize=True) * 100
    }).fillna(0)
    cmp.plot(kind='bar', ax=ax, width=0.75, alpha=0.85, edgecolor='black',
             color=['#3498db', '#e74c3c'])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentiment'); ax.set_ylabel('Percentage (%)')
    ax.legend(title='Platform'); ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_platform_sentiment_comparison.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 05_platform_sentiment_comparison.png")
plt.close()

# --- 06: Temporal sentiment trend (NEW) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Monthly Mean Sentiment Trend by Stance', fontsize=15, fontweight='bold')

for ax, df, lcol, title in [
        (axes[0], reddit_df,  REDDIT_LABEL_COL,  'Reddit – Monthly VADER Compound'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube – Monthly VADER Compound')]:
    if 'year_month' in df.columns and lcol in df.columns:
        t_data = (df[df[lcol].isin(['P', 'I', 'N'])]
                  .groupby(['year_month', lcol])['vader_compound']
                  .mean().unstack())
        t_data.index = t_data.index.astype(str)
        for stance in [c for c in ['P', 'I', 'N'] if c in t_data.columns]:
            ax.plot(t_data.index, t_data[stance], marker='o',
                    label=stance, color=palette.get(stance), linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Month'); ax.set_ylabel('Mean VADER Compound')
        ax.legend(title='Stance'); ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_temporal_sentiment_trend.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 06_temporal_sentiment_trend.png")
plt.close()

# --- 07: Sentiment by subreddit (NEW) ---
if 'subreddit' in reddit_df.columns:
    top_subs = reddit_df['subreddit'].value_counts().head(15).index
    sub_sent = (reddit_df[reddit_df['subreddit'].isin(top_subs)]
                .groupby('subreddit')['vader_compound'].mean()
                .sort_values())

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = ['#e74c3c' if v < 0 else '#27ae60' for v in sub_sent.values]
    ax.barh(sub_sent.index, sub_sent.values, color=bar_colors, alpha=0.85, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_title('Reddit – Mean VADER Sentiment by Subreddit (Top 15)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Compound Score'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_sentiment_by_subreddit.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 07_sentiment_by_subreddit.png")
    plt.close()

# --- 08: Subjectivity by stance (violin) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df, lcol, title in [
        (axes[0], reddit_df,  REDDIT_LABEL_COL,  'Reddit – Subjectivity by Stance'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube – Subjectivity by Stance')]:
    if lcol in df.columns:
        valid = df[df[lcol].isin(['P', 'I', 'N'])]
        sns.violinplot(data=valid, x=lcol, y='textblob_subjectivity',
                       palette=palette, ax=ax, order=['P', 'I', 'N'], inner='box')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Stance'); ax.set_ylabel('TextBlob Subjectivity')
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_subjectivity_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 08_subjectivity_by_stance.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

reddit_df.to_csv(OUTPUT_DIR / 'reddit_with_sentiment.csv', index=False, encoding='utf-8')
youtube_df.to_csv(OUTPUT_DIR / 'youtube_with_sentiment.csv', index=False, encoding='utf-8')
print("✔ Saved: reddit_with_sentiment.csv")
print("✔ Saved: youtube_with_sentiment.csv")

with open(OUTPUT_DIR / 'sentiment_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\nSENTIMENT ANALYSIS SUMMARY REPORT\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                             ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
        f.write(f"{name.upper()} VADER DISTRIBUTION:\n")
        f.write(str(df['vader_label'].value_counts()) + "\n\n")
        f.write(f"{name.upper()} VADER BY STANCE:\n")
        if lcol in df.columns:
            f.write(str(df.groupby(lcol)['vader_compound']
                        .agg(['mean', 'median', 'std']).round(4)) + "\n\n")
        f.write(f"{name.upper()} TEXTBLOB SUBJECTIVITY BY STANCE:\n")
        if lcol in df.columns:
            f.write(str(df.groupby(lcol)['textblob_subjectivity']
                        .agg(['mean', 'std']).round(4)) + "\n\n")
    f.write("=" * 80 + "\n")

print("✔ Saved: sentiment_summary_report.txt")

print("\n" + "=" * 80)
print("✅ SENTIMENT ANALYSIS COMPLETE!")
print("=" * 80)
