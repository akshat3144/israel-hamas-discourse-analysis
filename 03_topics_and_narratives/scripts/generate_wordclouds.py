"""
Word Cloud Generation for Israel-Hamas War Discourse
Generates word clouds by stance and sentiment for both platforms.
Column schema: Reddit uses self_text + Label; YouTube uses text + Label.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '03_topics_and_narratives' / 'outputs' / 'wordclouds'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_TEXT_COL   = 'self_text'
YOUTUBE_TEXT_COL  = 'text'
REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

print("=" * 70)
print("WORD CLOUD GENERATION")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading data...")
reddit_df  = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')
youtube_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'youtube_with_sentiment.csv')
print(f"✔ Reddit:  {len(reddit_df):,} rows")
print(f"✔ YouTube: {len(youtube_df):,} rows")

# ============================================================================
# TEXT CLEANING
# ============================================================================
def clean_text_for_wordcloud(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================================
# CUSTOM STOPWORDS
# ============================================================================
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    'israel', 'palestine', 'palestinian', 'israeli', 'hamas', 'gaza',
    'one', 'will', 'said', 'also', 'would', 'could', 'like', 'get', 'got',
    'even', 'make', 'made', 'take', 'took', 'know', 'well', 'really',
    'thing', 'things', 'say', 'saying', 'go', 'going', 'want', 'people',
    'just', 'much', 'many', 'time', 'war', 'conflict',
])

# ============================================================================
# HELPER: Generate and save a word cloud
# ============================================================================
def make_wordcloud(text, colormap='viridis', background='white',
                   max_words=120, width=900, height=600):
    if not text.strip():
        return None
    wc = WordCloud(
        width=width, height=height, background_color=background,
        stopwords=custom_stopwords, colormap=colormap,
        max_words=max_words, relative_scaling=0.45, min_font_size=9,
    )
    wc.generate(text)
    return wc

stances = [
    ('P', 'Pro-Palestine', '#2ecc71', 'Greens'),
    ('I', 'Pro-Israel',    '#3498db', 'Blues'),
    ('N', 'Neutral',       '#95a5a6', 'Greys'),
]

sentiments = [
    ('positive', 'Positive', '#27ae60', 'YlGn'),
    ('negative', 'Negative', '#e74c3c', 'Reds'),
    ('neutral',  'Neutral',  '#95a5a6', 'Greys'),
]

# ============================================================================
# 1. WORD CLOUDS BY STANCE — REDDIT
# ============================================================================
print("\n🎨 Reddit word clouds by stance...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Reddit: Word Clouds by Stance', fontsize=16, fontweight='bold', y=1.02)

for idx, (code, name, color, cmap) in enumerate(stances):
    sub = reddit_df[reddit_df[REDDIT_LABEL_COL] == code]
    text = ' '.join(sub[REDDIT_TEXT_COL].fillna('').apply(clean_text_for_wordcloud))
    wc = make_wordcloud(text, colormap=cmap)
    if wc:
        axes[idx].imshow(wc, interpolation='bilinear')
    else:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=18)
    axes[idx].set_title(f'{name}\n(n={len(sub):,})', fontsize=13, fontweight='bold', color=color)
    axes[idx].axis('off')
    print(f"  ✔ {name}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_reddit_wordclouds_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 01_reddit_wordclouds_by_stance.png")
plt.close()

# ============================================================================
# 2. WORD CLOUDS BY STANCE — YOUTUBE
# ============================================================================
print("\n🎨 YouTube word clouds by stance...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('YouTube: Word Clouds by Stance', fontsize=16, fontweight='bold', y=1.02)

for idx, (code, name, color, cmap) in enumerate(stances):
    sub = youtube_df[youtube_df[YOUTUBE_LABEL_COL] == code]
    text = ' '.join(sub[YOUTUBE_TEXT_COL].fillna('').apply(clean_text_for_wordcloud))
    wc = make_wordcloud(text, colormap=cmap)
    if wc:
        axes[idx].imshow(wc, interpolation='bilinear')
    else:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=18)
    axes[idx].set_title(f'{name}\n(n={len(sub):,})', fontsize=13, fontweight='bold', color=color)
    axes[idx].axis('off')
    print(f"  ✔ {name}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_youtube_wordclouds_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 02_youtube_wordclouds_by_stance.png")
plt.close()

# ============================================================================
# 3. WORD CLOUDS BY SENTIMENT — REDDIT
# ============================================================================
print("\n🎨 Reddit word clouds by sentiment...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Reddit: Word Clouds by Sentiment (VADER)', fontsize=16, fontweight='bold', y=1.02)

for idx, (sent_code, sent_name, color, cmap) in enumerate(sentiments):
    sub = reddit_df[reddit_df['vader_label'] == sent_code]
    text = ' '.join(sub[REDDIT_TEXT_COL].fillna('').apply(clean_text_for_wordcloud))
    wc = make_wordcloud(text, colormap=cmap)
    if wc:
        axes[idx].imshow(wc, interpolation='bilinear')
    else:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=18)
    axes[idx].set_title(f'{sent_name}\n(n={len(sub):,})', fontsize=13, fontweight='bold', color=color)
    axes[idx].axis('off')
    print(f"  ✔ {sent_name}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_reddit_wordclouds_by_sentiment.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 03_reddit_wordclouds_by_sentiment.png")
plt.close()

# ============================================================================
# 4. WORD CLOUDS BY SENTIMENT — YOUTUBE
# ============================================================================
print("\n🎨 YouTube word clouds by sentiment...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('YouTube: Word Clouds by Sentiment (VADER)', fontsize=16, fontweight='bold', y=1.02)

for idx, (sent_code, sent_name, color, cmap) in enumerate(sentiments):
    sub = youtube_df[youtube_df['vader_label'] == sent_code]
    text = ' '.join(sub[YOUTUBE_TEXT_COL].fillna('').apply(clean_text_for_wordcloud))
    wc = make_wordcloud(text, colormap=cmap)
    if wc:
        axes[idx].imshow(wc, interpolation='bilinear')
    else:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=18)
    axes[idx].set_title(f'{sent_name}\n(n={len(sub):,})', fontsize=13, fontweight='bold', color=color)
    axes[idx].axis('off')
    print(f"  ✔ {sent_name}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_youtube_wordclouds_by_sentiment.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 04_youtube_wordclouds_by_sentiment.png")
plt.close()

# ============================================================================
# 5. PLATFORM COMPARISON — OVERALL
# ============================================================================
print("\n🎨 Platform comparison word clouds...")
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle('Platform Comparison: Overall Word Clouds', fontsize=16, fontweight='bold')

r_all = ' '.join(reddit_df[REDDIT_TEXT_COL].fillna('').apply(clean_text_for_wordcloud))
y_all = ' '.join(youtube_df[YOUTUBE_TEXT_COL].fillna('').apply(clean_text_for_wordcloud))

wc_r = make_wordcloud(r_all, colormap='Blues', max_words=150)
wc_y = make_wordcloud(y_all, colormap='Reds',  max_words=150)

if wc_r:
    axes[0].imshow(wc_r, interpolation='bilinear')
axes[0].set_title(f'Reddit (n={len(reddit_df):,})', fontsize=14, fontweight='bold', color='#3498db')
axes[0].axis('off')

if wc_y:
    axes[1].imshow(wc_y, interpolation='bilinear')
axes[1].set_title(f'YouTube (n={len(youtube_df):,})', fontsize=14, fontweight='bold', color='#e74c3c')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_platform_comparison_wordclouds.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 05_platform_comparison_wordclouds.png")
plt.close()

# ============================================================================
# 6. SAVE WORD FREQUENCY REPORT
# ============================================================================
def get_top_words(text_series, n=20):
    all_text = ' '.join(text_series.apply(clean_text_for_wordcloud))
    words = [w for w in all_text.split()
             if w not in custom_stopwords and len(w) > 3]
    return Counter(words).most_common(n)

reddit_results  = {}
youtube_results = {}

for code, name, _, _ in stances:
    sub_r = reddit_df[reddit_df[REDDIT_LABEL_COL] == code][REDDIT_TEXT_COL].fillna('')
    reddit_results[name] = get_top_words(sub_r, 15)
    sub_y = youtube_df[youtube_df[YOUTUBE_LABEL_COL] == code][YOUTUBE_TEXT_COL].fillna('')
    youtube_results[name] = get_top_words(sub_y, 15)

with open(OUTPUT_DIR / 'word_frequency_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\nWORD CLOUD ANALYSIS REPORT\n")
    f.write("Top Words by Stance and Platform\n" + "=" * 70 + "\n\n")

    f.write("REDDIT TOP WORDS BY STANCE\n" + "-" * 70 + "\n")
    for name, top_words in reddit_results.items():
        f.write(f"\n{name}:\n")
        for word, count in top_words:
            f.write(f"  {word:.<30} {count:>6}\n")

    f.write("\n" + "=" * 70 + "\n\nYOUTUBE TOP WORDS BY STANCE\n" + "-" * 70 + "\n")
    for name, top_words in youtube_results.items():
        f.write(f"\n{name}:\n")
        for word, count in top_words:
            f.write(f"  {word:.<30} {count:>6}\n")

print("✔ Saved: word_frequency_report.txt")

print("\n" + "=" * 70)
print("✅ WORD CLOUD GENERATION COMPLETE")
print("=" * 70)
