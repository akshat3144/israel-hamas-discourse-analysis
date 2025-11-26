"""
Sentiment Analysis for Israel-Hamas War Discourse
Analyzes sentiment patterns across Reddit and YouTube data
Uses multiple approaches: VADER, TextBlob, and Transformers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Optional: Transformers for more advanced sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: pip install transformers torch")

import os
if not os.path.exists('sentiment_output'):
    os.makedirs('sentiment_output')

print("=" * 80)
print("SENTIMENT ANALYSIS - ISRAEL-HAMAS WAR DISCOURSE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading processed data...")
reddit_df = pd.read_csv('eda_output/reddit_processed.csv')
youtube_df = pd.read_csv('eda_output/youtube_processed.csv')

print(f"✓ Reddit: {len(reddit_df):,} rows")
print(f"✓ YouTube: {len(youtube_df):,} rows")

# ============================================================================
# INITIALIZE SENTIMENT ANALYZERS
# ============================================================================
print("\n🔧 Initializing sentiment analyzers...")
vader = SentimentIntensityAnalyzer()
print("✓ VADER Sentiment Analyzer loaded")
print("✓ TextBlob Sentiment Analyzer loaded")

# ============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================================================

def get_vader_sentiment(text):
    """Get VADER sentiment scores"""
    if pd.isna(text) or text == '':
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0, 'label': 'neutral'}
    
    scores = vader.polarity_scores(str(text))
    
    # Classify based on compound score
    if scores['compound'] >= 0.05:
        label = 'positive'
    elif scores['compound'] <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    scores['label'] = label
    return scores


def get_textblob_sentiment(text):
    """Get TextBlob sentiment scores"""
    if pd.isna(text) or text == '':
        return {'polarity': 0, 'subjectivity': 0, 'label': 'neutral'}
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Classify based on polarity
    if polarity > 0.1:
        label = 'positive'
    elif polarity < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'label': label
    }


def classify_sentiment(compound_score):
    """Classify sentiment based on compound score"""
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# ============================================================================
# ANALYZE REDDIT DATA
# ============================================================================
print("\n" + "=" * 80)
print("ANALYZING REDDIT SENTIMENT")
print("=" * 80)

# Determine text column
reddit_text_col = 'self_text' if 'self_text' in reddit_df.columns else 'clean_text_comments'
reddit_label_col = 'Label' if 'Label' in reddit_df.columns else 'label'

print(f"\nAnalyzing sentiment for Reddit comments using '{reddit_text_col}' column...")

# VADER Sentiment
print("\n🔍 Running VADER sentiment analysis...")
reddit_vader = reddit_df[reddit_text_col].fillna('').astype(str).apply(get_vader_sentiment)
reddit_df['vader_compound'] = reddit_vader.apply(lambda x: x['compound'])
reddit_df['vader_pos'] = reddit_vader.apply(lambda x: x['pos'])
reddit_df['vader_neu'] = reddit_vader.apply(lambda x: x['neu'])
reddit_df['vader_neg'] = reddit_vader.apply(lambda x: x['neg'])
reddit_df['vader_label'] = reddit_vader.apply(lambda x: x['label'])
print("✓ VADER analysis complete")

# TextBlob Sentiment
print("\n🔍 Running TextBlob sentiment analysis...")
reddit_textblob = reddit_df[reddit_text_col].fillna('').astype(str).apply(get_textblob_sentiment)
reddit_df['textblob_polarity'] = reddit_textblob.apply(lambda x: x['polarity'])
reddit_df['textblob_subjectivity'] = reddit_textblob.apply(lambda x: x['subjectivity'])
reddit_df['textblob_label'] = reddit_textblob.apply(lambda x: x['label'])
print("✓ TextBlob analysis complete")

# Display results
print("\n📊 REDDIT SENTIMENT DISTRIBUTION (VADER):")
print(reddit_df['vader_label'].value_counts())
print("\nPercentages:")
print(reddit_df['vader_label'].value_counts(normalize=True) * 100)

print("\n📊 REDDIT SENTIMENT DISTRIBUTION (TextBlob):")
print(reddit_df['textblob_label'].value_counts())
print("\nPercentages:")
print(reddit_df['textblob_label'].value_counts(normalize=True) * 100)

# ============================================================================
# ANALYZE YOUTUBE DATA
# ============================================================================
print("\n" + "=" * 80)
print("ANALYZING YOUTUBE SENTIMENT")
print("=" * 80)

youtube_text_col = 'text'
youtube_label_col = 'label' if 'label' in youtube_df.columns else 'Label'

print(f"\nAnalyzing sentiment for YouTube comments using '{youtube_text_col}' column...")

# VADER Sentiment
print("\n🔍 Running VADER sentiment analysis...")
youtube_vader = youtube_df[youtube_text_col].fillna('').astype(str).apply(get_vader_sentiment)
youtube_df['vader_compound'] = youtube_vader.apply(lambda x: x['compound'])
youtube_df['vader_pos'] = youtube_vader.apply(lambda x: x['pos'])
youtube_df['vader_neu'] = youtube_vader.apply(lambda x: x['neu'])
youtube_df['vader_neg'] = youtube_vader.apply(lambda x: x['neg'])
youtube_df['vader_label'] = youtube_vader.apply(lambda x: x['label'])
print("✓ VADER analysis complete")

# TextBlob Sentiment
print("\n🔍 Running TextBlob sentiment analysis...")
youtube_textblob = youtube_df[youtube_text_col].fillna('').astype(str).apply(get_textblob_sentiment)
youtube_df['textblob_polarity'] = youtube_textblob.apply(lambda x: x['polarity'])
youtube_df['textblob_subjectivity'] = youtube_textblob.apply(lambda x: x['subjectivity'])
youtube_df['textblob_label'] = youtube_textblob.apply(lambda x: x['label'])
print("✓ TextBlob analysis complete")

# Display results
print("\n📊 YOUTUBE SENTIMENT DISTRIBUTION (VADER):")
print(youtube_df['vader_label'].value_counts())
print("\nPercentages:")
print(youtube_df['vader_label'].value_counts(normalize=True) * 100)

print("\n📊 YOUTUBE SENTIMENT DISTRIBUTION (TextBlob):")
print(youtube_df['textblob_label'].value_counts())
print("\nPercentages:")
print(youtube_df['textblob_label'].value_counts(normalize=True) * 100)

# ============================================================================
# SENTIMENT BY STANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SENTIMENT BY STANCE ANALYSIS")
print("=" * 80)

# Reddit sentiment by stance
print("\n📊 REDDIT - Average Sentiment by Stance (VADER):")
reddit_sentiment_by_stance = reddit_df.groupby(reddit_label_col).agg({
    'vader_compound': ['mean', 'median', 'std'],
    'vader_pos': 'mean',
    'vader_neg': 'mean',
    'vader_neu': 'mean'
}).round(3)
print(reddit_sentiment_by_stance)

print("\n📊 REDDIT - Average Sentiment by Stance (TextBlob):")
reddit_textblob_by_stance = reddit_df.groupby(reddit_label_col).agg({
    'textblob_polarity': ['mean', 'median', 'std'],
    'textblob_subjectivity': ['mean', 'std']
}).round(3)
print(reddit_textblob_by_stance)

# YouTube sentiment by stance
print("\n📊 YOUTUBE - Average Sentiment by Stance (VADER):")
youtube_sentiment_by_stance = youtube_df.groupby(youtube_label_col).agg({
    'vader_compound': ['mean', 'median', 'std'],
    'vader_pos': 'mean',
    'vader_neg': 'mean',
    'vader_neu': 'mean'
}).round(3)
print(youtube_sentiment_by_stance)

print("\n📊 YOUTUBE - Average Sentiment by Stance (TextBlob):")
youtube_textblob_by_stance = youtube_df.groupby(youtube_label_col).agg({
    'textblob_polarity': ['mean', 'median', 'std'],
    'textblob_subjectivity': ['mean', 'std']
}).round(3)
print(youtube_textblob_by_stance)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 1. Sentiment Distribution by Platform
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Reddit VADER
reddit_vader_counts = reddit_df['vader_label'].value_counts()
colors_sent = ['#2ecc71', '#95a5a6', '#e74c3c']
axes[0, 0].bar(reddit_vader_counts.index, reddit_vader_counts.values, color=colors_sent, alpha=0.8, edgecolor='black')
axes[0, 0].set_title('Reddit Sentiment Distribution (VADER)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(reddit_vader_counts.values):
    axes[0, 0].text(i, v + max(reddit_vader_counts.values)*0.02, f'{v}\n({v/reddit_vader_counts.sum()*100:.1f}%)', 
                    ha='center', fontweight='bold')

# YouTube VADER
youtube_vader_counts = youtube_df['vader_label'].value_counts()
axes[0, 1].bar(youtube_vader_counts.index, youtube_vader_counts.values, color=colors_sent, alpha=0.8, edgecolor='black')
axes[0, 1].set_title('YouTube Sentiment Distribution (VADER)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Sentiment')
axes[0, 1].set_ylabel('Count')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(youtube_vader_counts.values):
    axes[0, 1].text(i, v + max(youtube_vader_counts.values)*0.02, f'{v}\n({v/youtube_vader_counts.sum()*100:.1f}%)', 
                    ha='center', fontweight='bold')

# Reddit TextBlob
reddit_tb_counts = reddit_df['textblob_label'].value_counts()
axes[1, 0].bar(reddit_tb_counts.index, reddit_tb_counts.values, color=colors_sent, alpha=0.8, edgecolor='black')
axes[1, 0].set_title('Reddit Sentiment Distribution (TextBlob)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Sentiment')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(reddit_tb_counts.values):
    axes[1, 0].text(i, v + max(reddit_tb_counts.values)*0.02, f'{v}\n({v/reddit_tb_counts.sum()*100:.1f}%)', 
                    ha='center', fontweight='bold')

# YouTube TextBlob
youtube_tb_counts = youtube_df['textblob_label'].value_counts()
axes[1, 1].bar(youtube_tb_counts.index, youtube_tb_counts.values, color=colors_sent, alpha=0.8, edgecolor='black')
axes[1, 1].set_title('YouTube Sentiment Distribution (TextBlob)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Sentiment')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(youtube_tb_counts.values):
    axes[1, 1].text(i, v + max(youtube_tb_counts.values)*0.02, f'{v}\n({v/youtube_tb_counts.sum()*100:.1f}%)', 
                    ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_output/01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_sentiment_distribution.png")
plt.close()

# 2. Sentiment by Stance - Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reddit
reddit_stance_sent = pd.crosstab(reddit_df[reddit_label_col], reddit_df['vader_label'], normalize='index') * 100
sns.heatmap(reddit_stance_sent, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0], cbar_kws={'label': 'Percentage (%)'})
axes[0].set_title('Reddit: Sentiment Distribution by Stance (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Stance')

# YouTube
youtube_stance_sent = pd.crosstab(youtube_df[youtube_label_col], youtube_df['vader_label'], normalize='index') * 100
sns.heatmap(youtube_stance_sent, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('YouTube: Sentiment Distribution by Stance (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Stance')

plt.tight_layout()
plt.savefig('sentiment_output/02_sentiment_by_stance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_sentiment_by_stance_heatmap.png")
plt.close()

# 3. Compound Score Distribution by Stance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reddit
sns.boxplot(data=reddit_df, x=reddit_label_col, y='vader_compound', ax=axes[0])
axes[0].set_title('Reddit: VADER Compound Score by Stance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Stance')
axes[0].set_ylabel('VADER Compound Score')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0].grid(axis='y', alpha=0.3)

# YouTube
sns.boxplot(data=youtube_df, x=youtube_label_col, y='vader_compound', ax=axes[1])
axes[1].set_title('YouTube: VADER Compound Score by Stance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Stance')
axes[1].set_ylabel('VADER Compound Score')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_output/03_compound_score_by_stance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_compound_score_by_stance.png")
plt.close()

# 4. Polarity vs Subjectivity Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reddit
stance_colors = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}
for stance in reddit_df[reddit_label_col].unique():
    if pd.notna(stance):
        subset = reddit_df[reddit_df[reddit_label_col] == stance]
        axes[0].scatter(subset['textblob_subjectivity'], subset['textblob_polarity'], 
                       alpha=0.5, label=stance, c=stance_colors.get(stance, '#95a5a6'), s=30)
axes[0].set_title('Reddit: Polarity vs Subjectivity by Stance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Subjectivity')
axes[0].set_ylabel('Polarity')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0].axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
axes[0].legend()
axes[0].grid(alpha=0.3)

# YouTube
for stance in youtube_df[youtube_label_col].unique():
    if pd.notna(stance):
        subset = youtube_df[youtube_df[youtube_label_col] == stance]
        axes[1].scatter(subset['textblob_subjectivity'], subset['textblob_polarity'], 
                       alpha=0.5, label=stance, c=stance_colors.get(stance, '#95a5a6'), s=30)
axes[1].set_title('YouTube: Polarity vs Subjectivity by Stance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Subjectivity')
axes[1].set_ylabel('Polarity')
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[1].axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_output/04_polarity_subjectivity_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_polarity_subjectivity_scatter.png")
plt.close()

# 5. Platform Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# VADER comparison
vader_comparison = pd.DataFrame({
    'Reddit': reddit_df['vader_label'].value_counts(normalize=True) * 100,
    'YouTube': youtube_df['vader_label'].value_counts(normalize=True) * 100
}).fillna(0)

vader_comparison.plot(kind='bar', ax=axes[0], width=0.8, alpha=0.8, edgecolor='black')
axes[0].set_title('Sentiment Comparison: Reddit vs YouTube (VADER)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(title='Platform')
axes[0].grid(axis='y', alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.1f%%', padding=3)

# TextBlob comparison
textblob_comparison = pd.DataFrame({
    'Reddit': reddit_df['textblob_label'].value_counts(normalize=True) * 100,
    'YouTube': youtube_df['textblob_label'].value_counts(normalize=True) * 100
}).fillna(0)

textblob_comparison.plot(kind='bar', ax=axes[1], width=0.8, alpha=0.8, edgecolor='black')
axes[1].set_title('Sentiment Comparison: Reddit vs YouTube (TextBlob)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(title='Platform')
axes[1].grid(axis='y', alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig('sentiment_output/05_platform_sentiment_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_platform_sentiment_comparison.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save enhanced datasets
reddit_df.to_csv('sentiment_output/reddit_with_sentiment.csv', index=False, encoding='utf-8')
youtube_df.to_csv('sentiment_output/youtube_with_sentiment.csv', index=False, encoding='utf-8')
print("✓ Saved: reddit_with_sentiment.csv")
print("✓ Saved: youtube_with_sentiment.csv")

# Create summary report
with open('sentiment_output/sentiment_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SENTIMENT ANALYSIS SUMMARY REPORT\n")
    f.write("Israel-Hamas War Discourse Analysis\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. REDDIT SENTIMENT DISTRIBUTION (VADER)\n")
    f.write("-" * 80 + "\n")
    f.write(str(reddit_df['vader_label'].value_counts()) + "\n\n")
    f.write("Percentages:\n")
    f.write(str(reddit_df['vader_label'].value_counts(normalize=True) * 100) + "\n\n")
    
    f.write("2. YOUTUBE SENTIMENT DISTRIBUTION (VADER)\n")
    f.write("-" * 80 + "\n")
    f.write(str(youtube_df['vader_label'].value_counts()) + "\n\n")
    f.write("Percentages:\n")
    f.write(str(youtube_df['vader_label'].value_counts(normalize=True) * 100) + "\n\n")
    
    f.write("3. SENTIMENT BY STANCE - REDDIT (VADER)\n")
    f.write("-" * 80 + "\n")
    f.write(str(reddit_sentiment_by_stance) + "\n\n")
    
    f.write("4. SENTIMENT BY STANCE - YOUTUBE (VADER)\n")
    f.write("-" * 80 + "\n")
    f.write(str(youtube_sentiment_by_stance) + "\n\n")
    
    f.write("=" * 80 + "\n")

print("✓ Saved: sentiment_summary_report.txt")

print("\n" + "=" * 80)
print("✅ SENTIMENT ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll outputs saved to: sentiment_output/")
print("\nGenerated files:")
print("  - 01_sentiment_distribution.png")
print("  - 02_sentiment_by_stance_heatmap.png")
print("  - 03_compound_score_by_stance.png")
print("  - 04_polarity_subjectivity_scatter.png")
print("  - 05_platform_sentiment_comparison.png")
print("  - reddit_with_sentiment.csv")
print("  - youtube_with_sentiment.csv")
print("  - sentiment_summary_report.txt")
print("\n" + "=" * 80)
