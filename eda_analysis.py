"""
Exploratory Data Analysis (EDA) for Israel-Hamas War Discourse
Analyzes Reddit and YouTube data to understand discourse patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory for plots
import os
if not os.path.exists('eda_output'):
    os.makedirs('eda_output')

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - ISRAEL-HAMMAS WAR DISCOURSE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df = pd.read_excel('data/reddit_labeled.xlsx')
youtube_df = pd.read_excel('data/youtube_labeled.xlsx')

print(f"✓ Reddit data: {len(reddit_df)} rows")
print(f"✓ YouTube data: {len(youtube_df)} rows")

# Clean and convert data types
print("\n🔧 Converting data types...")
# Reddit numeric conversions
if 'score' in reddit_df.columns:
    reddit_df['score'] = pd.to_numeric(reddit_df['score'], errors='coerce')
if 'post_score' in reddit_df.columns:
    reddit_df['post_score'] = pd.to_numeric(reddit_df['post_score'], errors='coerce')
if 'post_upvote_ratio' in reddit_df.columns:
    reddit_df['post_upvote_ratio'] = pd.to_numeric(reddit_df['post_upvote_ratio'], errors='coerce')
if 'user_total_karma' in reddit_df.columns:
    reddit_df['user_total_karma'] = pd.to_numeric(reddit_df['user_total_karma'], errors='coerce')

# YouTube numeric conversions
if 'likeCount' in youtube_df.columns:
    youtube_df['likeCount'] = pd.to_numeric(youtube_df['likeCount'], errors='coerce')
if 'replyCount' in youtube_df.columns:
    youtube_df['replyCount'] = pd.to_numeric(youtube_df['replyCount'], errors='coerce')

print("✓ Data type conversion complete")

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 80)

# Basic info
print("\n--- REDDIT DATA ---")
print(f"Total posts/comments: {len(reddit_df)}")
print(f"Columns: {list(reddit_df.columns)}")
print(f"\nData types:")
print(reddit_df.dtypes)

print("\n--- YOUTUBE DATA ---")
print(f"Total comments: {len(youtube_df)}")
print(f"Columns: {list(youtube_df.columns)}")
print(f"\nData types:")
print(youtube_df.dtypes)

# Check for labeled data
print("\n--- LABEL DISTRIBUTION ---")
reddit_label_col = 'Label' if 'Label' in reddit_df.columns else 'label'
youtube_label_col = 'label' if 'label' in youtube_df.columns else 'Label'

print("\nReddit Stance Distribution:")
if reddit_label_col in reddit_df.columns:
    reddit_labels = reddit_df[reddit_label_col].value_counts()
    print(reddit_labels)
    print(f"\nPercentages:")
    print(reddit_df[reddit_label_col].value_counts(normalize=True) * 100)
else:
    print("⚠️  No label column found in Reddit data")

print("\nYouTube Stance Distribution:")
if youtube_label_col in youtube_df.columns:
    youtube_labels = youtube_df[youtube_label_col].value_counts()
    print(youtube_labels)
    print(f"\nPercentages:")
    print(youtube_df[youtube_label_col].value_counts(normalize=True) * 100)
else:
    print("⚠️  No label column found in YouTube data")

# ============================================================================
# 2. MISSING DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. MISSING DATA ANALYSIS")
print("=" * 80)

print("\n--- REDDIT MISSING DATA ---")
reddit_missing = reddit_df.isnull().sum()
reddit_missing_pct = (reddit_df.isnull().sum() / len(reddit_df)) * 100
reddit_missing_df = pd.DataFrame({
    'Missing Count': reddit_missing,
    'Percentage': reddit_missing_pct
})
print(reddit_missing_df[reddit_missing_df['Missing Count'] > 0])

print("\n--- YOUTUBE MISSING DATA ---")
youtube_missing = youtube_df.isnull().sum()
youtube_missing_pct = (youtube_df.isnull().sum() / len(youtube_df)) * 100
youtube_missing_df = pd.DataFrame({
    'Missing Count': youtube_missing,
    'Percentage': youtube_missing_pct
})
print(youtube_missing_df[youtube_missing_df['Missing Count'] > 0])

# ============================================================================
# 3. ENGAGEMENT METRICS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. ENGAGEMENT METRICS ANALYSIS")
print("=" * 80)

# Reddit engagement
print("\n--- REDDIT ENGAGEMENT METRICS ---")
if 'score' in reddit_df.columns:
    print(f"\nScore Statistics:")
    print(reddit_df['score'].describe())
    print(f"Median score: {reddit_df['score'].median()}")
    
if 'controversiality' in reddit_df.columns:
    print(f"\nControversiality Distribution:")
    print(reddit_df['controversiality'].value_counts())

if 'num_comments' in reddit_df.columns:
    print(f"\nNumber of Comments Statistics:")
    print(reddit_df['num_comments'].describe())

# YouTube engagement
print("\n--- YOUTUBE ENGAGEMENT METRICS ---")
if 'likeCount' in youtube_df.columns:
    print(f"\nLike Count Statistics:")
    print(youtube_df['likeCount'].describe())
    print(f"Median likes: {youtube_df['likeCount'].median()}")

if 'replyCount' in youtube_df.columns:
    print(f"\nReply Count Statistics:")
    print(youtube_df['replyCount'].describe())

# ============================================================================
# 4. CONTENT LENGTH ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. CONTENT LENGTH ANALYSIS")
print("=" * 80)

# Reddit text length
if 'self_text' in reddit_df.columns:
    reddit_df['text_length'] = reddit_df['self_text'].fillna('').astype(str).str.len()
    print("\n--- REDDIT TEXT LENGTH ---")
    print(reddit_df['text_length'].describe())
    print(f"Median length: {reddit_df['text_length'].median()}")

# YouTube text length
if 'text' in youtube_df.columns:
    youtube_df['text_length'] = youtube_df['text'].fillna('').astype(str).str.len()
    print("\n--- YOUTUBE TEXT LENGTH ---")
    print(youtube_df['text_length'].describe())
    print(f"Median length: {youtube_df['text_length'].median()}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("5. GENERATING VISUALIZATIONS")
print("=" * 80)

# Set up the plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 5.1 Stance Distribution Comparison
if reddit_label_col in reddit_df.columns and youtube_label_col in youtube_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reddit
    reddit_counts = reddit_df[reddit_label_col].value_counts()
    colors = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}
    reddit_colors = [colors.get(label, '#95a5a6') for label in reddit_counts.index]
    axes[0].bar(reddit_counts.index, reddit_counts.values, color=reddit_colors, alpha=0.8)
    axes[0].set_title('Reddit Stance Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Stance')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(reddit_counts.values):
        axes[0].text(i, v + max(reddit_counts.values)*0.02, str(v), ha='center', fontweight='bold')
    
    # YouTube
    youtube_counts = youtube_df[youtube_label_col].value_counts()
    youtube_colors = [colors.get(label, '#95a5a6') for label in youtube_counts.index]
    axes[1].bar(youtube_counts.index, youtube_counts.values, color=youtube_colors, alpha=0.8)
    axes[1].set_title('YouTube Stance Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Stance')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(youtube_counts.values):
        axes[1].text(i, v + max(youtube_counts.values)*0.02, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_output/01_stance_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_stance_distribution.png")
    plt.close()

# 5.2 Stance Distribution - Pie Charts
if reddit_label_col in reddit_df.columns and youtube_label_col in youtube_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors_pie = ['#2ecc71', '#3498db', '#95a5a6']
    explode = (0.05, 0.05, 0.05)
    
    # Reddit pie
    reddit_counts = reddit_df[reddit_label_col].value_counts()
    axes[0].pie(reddit_counts.values, labels=reddit_counts.index, autopct='%1.1f%%',
                colors=colors_pie, explode=explode, shadow=True, startangle=90)
    axes[0].set_title('Reddit Stance Distribution (%)', fontsize=14, fontweight='bold')
    
    # YouTube pie
    youtube_counts = youtube_df[youtube_label_col].value_counts()
    axes[1].pie(youtube_counts.values, labels=youtube_counts.index, autopct='%1.1f%%',
                colors=colors_pie, explode=explode, shadow=True, startangle=90)
    axes[1].set_title('YouTube Stance Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_output/02_stance_pie_charts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_stance_pie_charts.png")
    plt.close()

# 5.3 Engagement by Stance - Reddit
if 'score' in reddit_df.columns and reddit_label_col in reddit_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    reddit_df.boxplot(column='score', by=reddit_label_col, ax=axes[0])
    axes[0].set_title('Reddit Score Distribution by Stance')
    axes[0].set_xlabel('Stance')
    axes[0].set_ylabel('Score')
    axes[0].get_figure().suptitle('')
    
    # Violin plot
    sns.violinplot(data=reddit_df, x=reddit_label_col, y='score', ax=axes[1])
    axes[1].set_title('Reddit Score Distribution by Stance (Violin)')
    axes[1].set_xlabel('Stance')
    axes[1].set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('eda_output/03_reddit_engagement_by_stance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_reddit_engagement_by_stance.png")
    plt.close()

# 5.4 Engagement by Stance - YouTube
if 'likeCount' in youtube_df.columns and youtube_label_col in youtube_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    youtube_df.boxplot(column='likeCount', by=youtube_label_col, ax=axes[0])
    axes[0].set_title('YouTube Like Count Distribution by Stance')
    axes[0].set_xlabel('Stance')
    axes[0].set_ylabel('Like Count')
    axes[0].get_figure().suptitle('')
    
    # Violin plot
    sns.violinplot(data=youtube_df, x=youtube_label_col, y='likeCount', ax=axes[1])
    axes[1].set_title('YouTube Like Count Distribution by Stance (Violin)')
    axes[1].set_xlabel('Stance')
    axes[1].set_ylabel('Like Count')
    
    plt.tight_layout()
    plt.savefig('eda_output/04_youtube_engagement_by_stance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_youtube_engagement_by_stance.png")
    plt.close()

# 5.5 Text Length Distribution
if 'text_length' in reddit_df.columns or 'text_length' in youtube_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'text_length' in reddit_df.columns:
        axes[0].hist(reddit_df['text_length'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].set_title('Reddit Text Length Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Text Length (characters)')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(reddit_df['text_length'].median(), color='red', linestyle='--', 
                       label=f'Median: {reddit_df["text_length"].median():.0f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    if 'text_length' in youtube_df.columns:
        axes[1].hist(youtube_df['text_length'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1].set_title('YouTube Text Length Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Text Length (characters)')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(youtube_df['text_length'].median(), color='red', linestyle='--',
                       label=f'Median: {youtube_df["text_length"].median():.0f}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_output/05_text_length_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_text_length_distribution.png")
    plt.close()

# 5.6 Platform Comparison - Combined View
if reddit_label_col in reddit_df.columns and youtube_label_col in youtube_df.columns:
    # Prepare data for comparison
    reddit_stance_pct = reddit_df[reddit_label_col].value_counts(normalize=True) * 100
    youtube_stance_pct = youtube_df[youtube_label_col].value_counts(normalize=True) * 100
    
    comparison_df = pd.DataFrame({
        'Reddit': reddit_stance_pct,
        'YouTube': youtube_stance_pct
    }).fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
    ax.set_title('Stance Distribution Comparison: Reddit vs YouTube (%)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Stance', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(title='Platform', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    plt.tight_layout()
    plt.savefig('eda_output/06_platform_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_platform_comparison.png")
    plt.close()

# 5.7 Controversiality Analysis (Reddit)
if 'controversiality' in reddit_df.columns and reddit_label_col in reddit_df.columns:
    # Create cross-tabulation
    controversy_cross = pd.crosstab(reddit_df[reddit_label_col], 
                                   reddit_df['controversiality'], 
                                   normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    controversy_cross.plot(kind='bar', ax=ax, stacked=False)
    ax.set_title('Reddit Controversiality by Stance (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stance', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(title='Controversial', labels=['No', 'Yes'])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('eda_output/07_controversiality_by_stance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_controversiality_by_stance.png")
    plt.close()

# 5.8 Correlation Heatmap - Reddit
if 'score' in reddit_df.columns and 'controversiality' in reddit_df.columns:
    reddit_numeric = reddit_df.select_dtypes(include=[np.number])
    if len(reddit_numeric.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = reddit_numeric.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Reddit Metrics Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('eda_output/08_reddit_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 08_reddit_correlation_heatmap.png")
        plt.close()

# 5.9 Correlation Heatmap - YouTube
if 'likeCount' in youtube_df.columns:
    youtube_numeric = youtube_df.select_dtypes(include=[np.number])
    if len(youtube_numeric.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = youtube_numeric.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('YouTube Metrics Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('eda_output/09_youtube_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 09_youtube_correlation_heatmap.png")
        plt.close()

# ============================================================================
# 6. SUMMARY STATISTICS EXPORT
# ============================================================================
print("\n" + "=" * 80)
print("6. EXPORTING SUMMARY STATISTICS")
print("=" * 80)

# Create summary report
with open('eda_output/eda_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS SUMMARY REPORT\n")
    f.write("Israel-Hamas War Discourse Analysis\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    # Dataset Overview
    f.write("1. DATASET OVERVIEW\n")
    f.write("-" * 80 + "\n")
    f.write(f"Reddit Posts/Comments: {len(reddit_df)}\n")
    f.write(f"YouTube Comments: {len(youtube_df)}\n")
    f.write(f"Total Data Points: {len(reddit_df) + len(youtube_df)}\n\n")
    
    # Stance Distribution
    f.write("2. STANCE DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    f.write("\nReddit:\n")
    if reddit_label_col in reddit_df.columns:
        f.write(str(reddit_df[reddit_label_col].value_counts()) + "\n")
        f.write("\nPercentages:\n")
        f.write(str(reddit_df[reddit_label_col].value_counts(normalize=True) * 100) + "\n")
    
    f.write("\nYouTube:\n")
    if youtube_label_col in youtube_df.columns:
        f.write(str(youtube_df[youtube_label_col].value_counts()) + "\n")
        f.write("\nPercentages:\n")
        f.write(str(youtube_df[youtube_label_col].value_counts(normalize=True) * 100) + "\n")
    
    # Engagement Statistics
    f.write("\n3. ENGAGEMENT STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write("\nReddit Score Statistics:\n")
    if 'score' in reddit_df.columns:
        f.write(str(reddit_df['score'].describe()) + "\n")
    
    f.write("\nYouTube Like Count Statistics:\n")
    if 'likeCount' in youtube_df.columns:
        f.write(str(youtube_df['likeCount'].describe()) + "\n")
    
    # Content Length
    f.write("\n4. CONTENT LENGTH ANALYSIS\n")
    f.write("-" * 80 + "\n")
    if 'text_length' in reddit_df.columns:
        f.write("\nReddit Text Length:\n")
        f.write(str(reddit_df['text_length'].describe()) + "\n")
    
    if 'text_length' in youtube_df.columns:
        f.write("\nYouTube Text Length:\n")
        f.write(str(youtube_df['text_length'].describe()) + "\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print("✓ Saved: eda_summary_report.txt")

# Export processed data with text length
reddit_df.to_csv('eda_output/reddit_processed.csv', index=False, encoding='utf-8')
youtube_df.to_csv('eda_output/youtube_processed.csv', index=False, encoding='utf-8')
print("✓ Saved: reddit_processed.csv")
print("✓ Saved: youtube_processed.csv")

print("\n" + "=" * 80)
print("✅ EDA COMPLETE!")
print("=" * 80)
print(f"\nAll outputs saved to: eda_output/")
print("\nGenerated files:")
print("  - 01_stance_distribution.png")
print("  - 02_stance_pie_charts.png")
print("  - 03_reddit_engagement_by_stance.png")
print("  - 04_youtube_engagement_by_stance.png")
print("  - 05_text_length_distribution.png")
print("  - 06_platform_comparison.png")
print("  - 07_controversiality_by_stance.png")
print("  - 08_reddit_correlation_heatmap.png")
print("  - 09_youtube_correlation_heatmap.png")
print("  - eda_summary_report.txt")
print("  - reddit_processed.csv")
print("  - youtube_processed.csv")
print("\n" + "=" * 80)
