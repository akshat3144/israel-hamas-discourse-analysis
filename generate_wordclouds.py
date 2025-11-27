"""
Phase 3: Word Cloud Visualization
Generate word clouds for each stance (Pro-Palestine, Pro-Israel, Neutral) on both platforms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
import re
from collections import Counter

# Create output directory
os.makedirs('wordcloud_output', exist_ok=True)

print("="*70)
print("PHASE 3: WORD CLOUD GENERATION")
print("="*70)
print()

# ============================================================================
# 1. LOAD DATA WITH SENTIMENT SCORES
# ============================================================================
print("📂 Loading data with sentiment scores...")
reddit_df = pd.read_csv('sentiment_output/reddit_with_sentiment.csv')
youtube_df = pd.read_csv('sentiment_output/youtube_with_sentiment.csv')
print(f"✓ Reddit: {len(reddit_df)} rows")
print(f"✓ YouTube: {len(youtube_df)} rows")
print()

# ============================================================================
# 2. TEXT PREPROCESSING FUNCTION
# ============================================================================
def clean_text_for_wordcloud(text):
    """Clean and prepare text for word cloud generation"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ============================================================================
# 3. CUSTOM STOPWORDS
# ============================================================================
custom_stopwords = set(STOPWORDS)
# Add domain-specific stopwords
custom_stopwords.update([
    'israel', 'palestine', 'palestinian', 'israeli', 'hamas',
    'one', 'will', 'said', 'also', 'would', 'could', 'like',
    'get', 'got', 'even', 'make', 'made', 'take', 'took',
    'know', 'well', 'really', 'thing', 'things', 'say', 'saying',
    'go', 'going', 'want', 'people', 'just', 'much', 'many'
])

# ============================================================================
# 4. GENERATE WORD CLOUDS BY STANCE - REDDIT
# ============================================================================
print("🎨 Generating Reddit word clouds by stance...")
print()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Reddit: Word Clouds by Stance', fontsize=16, fontweight='bold', y=1.02)

stances = [
    ('P', 'Pro-Palestine', '#2ecc71'),
    ('I', 'Pro-Israel', '#3498db'),
    ('N', 'Neutral', '#95a5a6')
]

for idx, (stance_code, stance_name, color) in enumerate(stances):
    # Filter data by stance
    stance_data = reddit_df[reddit_df['Label'] == stance_code]
    
    # Combine all text for this stance
    text_col = 'clean_text_posts' if 'clean_text_posts' in reddit_df.columns else 'selftext'
    all_text = ' '.join(stance_data[text_col].fillna('').apply(clean_text_for_wordcloud))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='white',
        stopwords=custom_stopwords,
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(all_text)
    
    # Plot
    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].set_title(f'{stance_name}\n({len(stance_data)} posts)', 
                       fontsize=12, fontweight='bold', color=color)
    axes[idx].axis('off')
    
    print(f"✓ Generated Reddit {stance_name} word cloud")

plt.tight_layout()
plt.savefig('wordcloud_output/01_reddit_wordclouds_by_stance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_reddit_wordclouds_by_stance.png")
print()

# ============================================================================
# 5. GENERATE WORD CLOUDS BY STANCE - YOUTUBE
# ============================================================================
print("🎨 Generating YouTube word clouds by stance...")
print()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('YouTube: Word Clouds by Stance', fontsize=16, fontweight='bold', y=1.02)

for idx, (stance_code, stance_name, color) in enumerate(stances):
    # Filter data by stance
    label_col = 'label' if 'label' in youtube_df.columns else 'Label'
    stance_data = youtube_df[youtube_df[label_col] == stance_code]
    
    # Combine all text for this stance
    text_col = 'clean_text_comments' if 'clean_text_comments' in youtube_df.columns else 'text'
    all_text = ' '.join(stance_data[text_col].fillna('').apply(clean_text_for_wordcloud))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='white',
        stopwords=custom_stopwords,
        colormap='plasma',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(all_text)
    
    # Plot
    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].set_title(f'{stance_name}\n({len(stance_data)} comments)', 
                       fontsize=12, fontweight='bold', color=color)
    axes[idx].axis('off')
    
    print(f"✓ Generated YouTube {stance_name} word cloud")

plt.tight_layout()
plt.savefig('wordcloud_output/02_youtube_wordclouds_by_stance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_youtube_wordclouds_by_stance.png")
print()

# ============================================================================
# 6. SENTIMENT-BASED WORD CLOUDS - REDDIT
# ============================================================================
print("🎨 Generating Reddit word clouds by sentiment...")
print()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Reddit: Word Clouds by Sentiment (VADER)', fontsize=16, fontweight='bold', y=1.02)

sentiments = [
    ('Positive', '#27ae60'),
    ('Negative', '#e74c3c'),
    ('Neutral', '#95a5a6')
]

for idx, (sentiment, color) in enumerate(sentiments):
    # Filter data by sentiment
    sentiment_data = reddit_df[reddit_df['vader_label'] == sentiment]
    
    # Combine all text for this sentiment
    text_col = 'clean_text_posts' if 'clean_text_posts' in reddit_df.columns else 'selftext'
    all_text = ' '.join(sentiment_data[text_col].fillna('').apply(clean_text_for_wordcloud))
    
    # Generate word cloud only if we have text
    if len(all_text.strip()) > 0:
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            stopwords=custom_stopwords,
            colormap='RdYlGn' if sentiment == 'Positive' else 'RdYlGn_r' if sentiment == 'Negative' else 'Greys',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_text)
        
        # Plot
        axes[idx].imshow(wordcloud, interpolation='bilinear')
    else:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=20)
        
    axes[idx].set_title(f'{sentiment}\n({len(sentiment_data)} posts)', 
                       fontsize=12, fontweight='bold', color=color)
    axes[idx].axis('off')
    
    print(f"✓ Generated Reddit {sentiment} word cloud")

plt.tight_layout()
plt.savefig('wordcloud_output/03_reddit_wordclouds_by_sentiment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_reddit_wordclouds_by_sentiment.png")
print()

# ============================================================================
# 7. SENTIMENT-BASED WORD CLOUDS - YOUTUBE
# ============================================================================
print("🎨 Generating YouTube word clouds by sentiment...")
print()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('YouTube: Word Clouds by Sentiment (VADER)', fontsize=16, fontweight='bold', y=1.02)

for idx, (sentiment, color) in enumerate(sentiments):
    # Filter data by sentiment
    sentiment_data = youtube_df[youtube_df['vader_label'] == sentiment]
    
    # Combine all text for this sentiment
    text_col = 'clean_text_comments' if 'clean_text_comments' in youtube_df.columns else 'text'
    all_text = ' '.join(sentiment_data[text_col].fillna('').apply(clean_text_for_wordcloud))
    
    # Generate word cloud only if we have text
    if len(all_text.strip()) > 0:
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            stopwords=custom_stopwords,
            colormap='RdYlGn' if sentiment == 'Positive' else 'RdYlGn_r' if sentiment == 'Negative' else 'Greys',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_text)
        
        # Plot
        axes[idx].imshow(wordcloud, interpolation='bilinear')
    else:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=20)
        
    axes[idx].set_title(f'{sentiment}\n({len(sentiment_data)} comments)', 
                       fontsize=12, fontweight='bold', color=color)
    axes[idx].axis('off')
    
    print(f"✓ Generated YouTube {sentiment} word cloud")

plt.tight_layout()
plt.savefig('wordcloud_output/04_youtube_wordclouds_by_sentiment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_youtube_wordclouds_by_sentiment.png")
print()

# ============================================================================
# 8. PLATFORM COMPARISON - COMBINED WORD CLOUDS
# ============================================================================
print("🎨 Generating platform comparison word clouds...")
print()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Platform Comparison: Overall Word Clouds', fontsize=16, fontweight='bold', y=0.98)

# Reddit overall
text_col = 'clean_text_posts' if 'clean_text_posts' in reddit_df.columns else 'selftext'
reddit_all_text = ' '.join(reddit_df[text_col].fillna('').apply(clean_text_for_wordcloud))
reddit_wordcloud = WordCloud(
    width=800,
    height=800,
    background_color='white',
    stopwords=custom_stopwords,
    colormap='Blues',
    max_words=150,
    relative_scaling=0.5,
    min_font_size=10
).generate(reddit_all_text)

axes[0].imshow(reddit_wordcloud, interpolation='bilinear')
axes[0].set_title(f'Reddit (n={len(reddit_df)})', fontsize=14, fontweight='bold', color='#3498db')
axes[0].axis('off')

# YouTube overall
text_col = 'clean_text_comments' if 'clean_text_comments' in youtube_df.columns else 'text'
youtube_all_text = ' '.join(youtube_df[text_col].fillna('').apply(clean_text_for_wordcloud))
youtube_wordcloud = WordCloud(
    width=800,
    height=800,
    background_color='white',
    stopwords=custom_stopwords,
    colormap='Reds',
    max_words=150,
    relative_scaling=0.5,
    min_font_size=10
).generate(youtube_all_text)

axes[1].imshow(youtube_wordcloud, interpolation='bilinear')
axes[1].set_title(f'YouTube (n={len(youtube_df)})', fontsize=14, fontweight='bold', color='#e74c3c')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('wordcloud_output/05_platform_comparison_wordclouds.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_platform_comparison_wordclouds.png")
print()

# ============================================================================
# 9. TOP WORDS EXTRACTION FOR REPORT
# ============================================================================
print("📊 Extracting top words for each category...")
print()

def get_top_words(text_series, n=20):
    """Extract top N words from text series"""
    all_text = ' '.join(text_series.apply(clean_text_for_wordcloud))
    words = [word for word in all_text.split() if word not in custom_stopwords and len(word) > 3]
    return Counter(words).most_common(n)

# Reddit by stance
reddit_results = {}
text_col_reddit = 'clean_text_posts' if 'clean_text_posts' in reddit_df.columns else 'selftext'
for stance_code, stance_name, _ in stances:
    stance_data = reddit_df[reddit_df['Label'] == stance_code][text_col_reddit].fillna('')
    top_words = get_top_words(stance_data, 15)
    reddit_results[stance_name] = top_words

# YouTube by stance
youtube_results = {}
text_col_youtube = 'clean_text_comments' if 'clean_text_comments' in youtube_df.columns else 'text'
label_col_youtube = 'label' if 'label' in youtube_df.columns else 'Label'
for stance_code, stance_name, _ in stances:
    stance_data = youtube_df[youtube_df[label_col_youtube] == stance_code][text_col_youtube].fillna('')
    top_words = get_top_words(stance_data, 15)
    youtube_results[stance_name] = top_words

# Save word frequency report
with open('wordcloud_output/word_frequency_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("WORD CLOUD ANALYSIS REPORT\n")
    f.write("Top Words by Stance and Platform\n")
    f.write("="*70 + "\n\n")
    
    f.write("REDDIT TOP WORDS BY STANCE\n")
    f.write("-"*70 + "\n")
    for stance_name, top_words in reddit_results.items():
        f.write(f"\n{stance_name}:\n")
        for word, count in top_words:
            f.write(f"  {word:.<30} {count:>5}\n")
    
    f.write("\n" + "="*70 + "\n\n")
    f.write("YOUTUBE TOP WORDS BY STANCE\n")
    f.write("-"*70 + "\n")
    for stance_name, top_words in youtube_results.items():
        f.write(f"\n{stance_name}:\n")
        for word, count in top_words:
            f.write(f"  {word:.<30} {count:>5}\n")

print("✓ Saved: word_frequency_report.txt")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print("✅ WORD CLOUD GENERATION COMPLETE")
print("="*70)
print()
print("📁 Generated Files:")
print("  - 01_reddit_wordclouds_by_stance.png (Pro-P, Pro-I, Neutral)")
print("  - 02_youtube_wordclouds_by_stance.png (Pro-P, Pro-I, Neutral)")
print("  - 03_reddit_wordclouds_by_sentiment.png (Positive, Negative, Neutral)")
print("  - 04_youtube_wordclouds_by_sentiment.png (Positive, Negative, Neutral)")
print("  - 05_platform_comparison_wordclouds.png (Reddit vs YouTube)")
print("  - word_frequency_report.txt (Top words by category)")
print()
print("📊 Total Visualizations: 5 (containing 13 word clouds)")
print()
