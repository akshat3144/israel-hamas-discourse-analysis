"""
Topic Modeling for Israel-Hamas War Discourse
Uses LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorization)
Identifies key themes and narratives in the discourse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import warnings
warnings.filterwarnings('ignore')

# For text preprocessing
import re
from collections import Counter

import os
if not os.path.exists('topic_modeling_output'):
    os.makedirs('topic_modeling_output')

print("=" * 80)
print("TOPIC MODELING - ISRAEL-HAMAS WAR DISCOURSE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading sentiment-enhanced data...")
reddit_df = pd.read_csv('sentiment_output/reddit_with_sentiment.csv')
youtube_df = pd.read_csv('sentiment_output/youtube_with_sentiment.csv')

print(f"✓ Reddit: {len(reddit_df):,} rows")
print(f"✓ YouTube: {len(youtube_df):,} rows")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
print("\n🔧 Preprocessing text data...")

# Identify text columns
reddit_text_col = 'self_text' if 'self_text' in reddit_df.columns else 'clean_text_comments'
youtube_text_col = 'text'
reddit_label_col = 'Label' if 'Label' in reddit_df.columns else 'label'
youtube_label_col = 'label' if 'label' in youtube_df.columns else 'Label'

# Clean text function
def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Apply cleaning
reddit_df['cleaned_text'] = reddit_df[reddit_text_col].apply(clean_text)
youtube_df['cleaned_text'] = youtube_df[youtube_text_col].apply(clean_text)

# Remove empty texts
reddit_df = reddit_df[reddit_df['cleaned_text'].str.len() > 20]
youtube_df = youtube_df[youtube_df['cleaned_text'].str.len() > 20]

print(f"✓ Cleaned Reddit texts: {len(reddit_df):,}")
print(f"✓ Cleaned YouTube texts: {len(youtube_df):,}")

# ============================================================================
# STOPWORDS
# ============================================================================
# Domain-specific stopwords
domain_stopwords = [
    'israel', 'israeli', 'palestine', 'palestinian', 'hamas', 'gaza', 'war',
    'conflict', 'just', 'like', 'people', 'know', 'think', 'going', 'said',
    'really', 'also', 'would', 'could', 'one', 'two', 'even', 'make', 'get',
    'want', 'see', 'say', 'tell', 'much', 'many', 'thing', 'way', 'time'
]

# ============================================================================
# WORD FREQUENCY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("WORD FREQUENCY ANALYSIS")
print("=" * 80)

def get_top_words(texts, n=20, stopwords=None):
    """Get top N words from texts"""
    vectorizer = CountVectorizer(max_features=n, stop_words='english')
    if stopwords:
        vectorizer = CountVectorizer(max_features=n, stop_words=list(set(list(vectorizer.get_stop_words()) + stopwords)))
    
    X = vectorizer.fit_transform(texts)
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
    return Counter(word_freq).most_common(n)

# Overall top words
print("\n📊 TOP 20 WORDS - REDDIT:")
reddit_top_words = get_top_words(reddit_df['cleaned_text'], n=20, stopwords=domain_stopwords)
for word, count in reddit_top_words:
    print(f"  {word}: {count:,}")

print("\n📊 TOP 20 WORDS - YOUTUBE:")
youtube_top_words = get_top_words(youtube_df['cleaned_text'], n=20, stopwords=domain_stopwords)
for word, count in youtube_top_words:
    print(f"  {word}: {count:,}")

# Top words by stance
print("\n📊 TOP WORDS BY STANCE - REDDIT:")
for stance in ['P', 'I', 'N']:
    if stance in reddit_df[reddit_label_col].unique():
        stance_texts = reddit_df[reddit_df[reddit_label_col] == stance]['cleaned_text']
        if len(stance_texts) > 0:
            print(f"\nStance {stance}:")
            top_words = get_top_words(stance_texts, n=10, stopwords=domain_stopwords)
            for word, count in top_words:
                print(f"  {word}: {count:,}")

# ============================================================================
# TOPIC MODELING - LDA
# ============================================================================
print("\n" + "=" * 80)
print("TOPIC MODELING - LATENT DIRICHLET ALLOCATION (LDA)")
print("=" * 80)

# Parameters
n_topics = 5
n_top_words = 10

# Reddit LDA
print("\n🔍 Running LDA on Reddit data...")
reddit_vectorizer = CountVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')
reddit_tf = reddit_vectorizer.fit_transform(reddit_df['cleaned_text'])

reddit_lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
reddit_lda.fit(reddit_tf)
print("✓ Reddit LDA complete")

# Display Reddit topics
print(f"\n📊 REDDIT - TOP {n_top_words} WORDS PER TOPIC:")
reddit_feature_names = reddit_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(reddit_lda.components_):
    top_words_idx = topic.argsort()[-n_top_words:][::-1]
    top_words = [reddit_feature_names[i] for i in top_words_idx]
    print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

# YouTube LDA
print("\n🔍 Running LDA on YouTube data...")
youtube_vectorizer = CountVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')
youtube_tf = youtube_vectorizer.fit_transform(youtube_df['cleaned_text'])

youtube_lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
youtube_lda.fit(youtube_tf)
print("✓ YouTube LDA complete")

# Display YouTube topics
print(f"\n📊 YOUTUBE - TOP {n_top_words} WORDS PER TOPIC:")
youtube_feature_names = youtube_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(youtube_lda.components_):
    top_words_idx = topic.argsort()[-n_top_words:][::-1]
    top_words = [youtube_feature_names[i] for i in top_words_idx]
    print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

# ============================================================================
# TOPIC MODELING - NMF
# ============================================================================
print("\n" + "=" * 80)
print("TOPIC MODELING - NON-NEGATIVE MATRIX FACTORIZATION (NMF)")
print("=" * 80)

# Reddit NMF
print("\n🔍 Running NMF on Reddit data...")
reddit_tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')
reddit_tfidf = reddit_tfidf_vectorizer.fit_transform(reddit_df['cleaned_text'])

reddit_nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
reddit_nmf.fit(reddit_tfidf)
print("✓ Reddit NMF complete")

# Display Reddit NMF topics
print(f"\n📊 REDDIT - TOP {n_top_words} WORDS PER TOPIC (NMF):")
reddit_tfidf_feature_names = reddit_tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(reddit_nmf.components_):
    top_words_idx = topic.argsort()[-n_top_words:][::-1]
    top_words = [reddit_tfidf_feature_names[i] for i in top_words_idx]
    print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

# YouTube NMF
print("\n🔍 Running NMF on YouTube data...")
youtube_tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')
youtube_tfidf = youtube_tfidf_vectorizer.fit_transform(youtube_df['cleaned_text'])

youtube_nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
youtube_nmf.fit(youtube_tfidf)
print("✓ YouTube NMF complete")

# Display YouTube NMF topics
print(f"\n📊 YOUTUBE - TOP {n_top_words} WORDS PER TOPIC (NMF):")
youtube_tfidf_feature_names = youtube_tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(youtube_nmf.components_):
    top_words_idx = topic.argsort()[-n_top_words:][::-1]
    top_words = [youtube_tfidf_feature_names[i] for i in top_words_idx]
    print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

plt.rcParams['figure.figsize'] = (12, 8)

# 1. Word Frequency Bar Charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reddit
reddit_words = [word for word, count in reddit_top_words]
reddit_counts = [count for word, count in reddit_top_words]
axes[0].barh(reddit_words, reddit_counts, color='#3498db', alpha=0.8, edgecolor='black')
axes[0].set_title('Reddit: Top 20 Most Frequent Words', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Frequency')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# YouTube
youtube_words = [word for word, count in youtube_top_words]
youtube_counts = [count for word, count in youtube_top_words]
axes[1].barh(youtube_words, youtube_counts, color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1].set_title('YouTube: Top 20 Most Frequent Words', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Frequency')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('topic_modeling_output/01_word_frequency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_word_frequency.png")
plt.close()

# 2. Topic Distribution Heatmaps - LDA
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reddit LDA heatmap
reddit_topic_word = reddit_lda.components_ / reddit_lda.components_.sum(axis=1)[:, np.newaxis]
top_words_per_topic = []
for topic in reddit_topic_word:
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words_per_topic.append([reddit_feature_names[i] for i in top_words_idx])

# Create simplified heatmap data
reddit_heatmap_data = []
for i, topic in enumerate(reddit_lda.components_):
    top_10_idx = topic.argsort()[-10:][::-1]
    reddit_heatmap_data.append(topic[top_10_idx])

sns.heatmap(reddit_heatmap_data, xticklabels=top_words_per_topic[0], 
            yticklabels=[f'Topic {i+1}' for i in range(n_topics)],
            cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Weight'})
axes[0].set_title('Reddit: LDA Topic-Word Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Top Words')
axes[0].set_ylabel('Topics')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# YouTube LDA heatmap
youtube_heatmap_data = []
youtube_top_words_per_topic = []
for i, topic in enumerate(youtube_lda.components_):
    top_10_idx = topic.argsort()[-10:][::-1]
    youtube_heatmap_data.append(topic[top_10_idx])
    youtube_top_words_per_topic.append([youtube_feature_names[i] for i in top_10_idx])

sns.heatmap(youtube_heatmap_data, xticklabels=youtube_top_words_per_topic[0],
            yticklabels=[f'Topic {i+1}' for i in range(n_topics)],
            cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Weight'})
axes[1].set_title('YouTube: LDA Topic-Word Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Top Words')
axes[1].set_ylabel('Topics')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('topic_modeling_output/02_lda_topic_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_lda_topic_heatmaps.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save topic modeling results
with open('topic_modeling_output/topic_modeling_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TOPIC MODELING REPORT\n")
    f.write("Israel-Hamas War Discourse Analysis\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. REDDIT - TOP 20 WORDS\n")
    f.write("-" * 80 + "\n")
    for word, count in reddit_top_words:
        f.write(f"{word}: {count:,}\n")
    
    f.write("\n2. YOUTUBE - TOP 20 WORDS\n")
    f.write("-" * 80 + "\n")
    for word, count in youtube_top_words:
        f.write(f"{word}: {count:,}\n")
    
    f.write(f"\n3. REDDIT - LDA TOPICS (Top {n_top_words} words per topic)\n")
    f.write("-" * 80 + "\n")
    for topic_idx, topic in enumerate(reddit_lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [reddit_feature_names[i] for i in top_words_idx]
        f.write(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}\n")
    
    f.write(f"\n4. YOUTUBE - LDA TOPICS (Top {n_top_words} words per topic)\n")
    f.write("-" * 80 + "\n")
    for topic_idx, topic in enumerate(youtube_lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [youtube_feature_names[i] for i in top_words_idx]
        f.write(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}\n")
    
    f.write(f"\n5. REDDIT - NMF TOPICS (Top {n_top_words} words per topic)\n")
    f.write("-" * 80 + "\n")
    for topic_idx, topic in enumerate(reddit_nmf.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [reddit_tfidf_feature_names[i] for i in top_words_idx]
        f.write(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}\n")
    
    f.write(f"\n6. YOUTUBE - NMF TOPICS (Top {n_top_words} words per topic)\n")
    f.write("-" * 80 + "\n")
    for topic_idx, topic in enumerate(youtube_nmf.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [youtube_tfidf_feature_names[i] for i in top_words_idx]
        f.write(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}\n")
    
    f.write("\n" + "=" * 80 + "\n")

print("✓ Saved: topic_modeling_report.txt")

print("\n" + "=" * 80)
print("✅ TOPIC MODELING COMPLETE!")
print("=" * 80)
print("\nAll outputs saved to: topic_modeling_output/")
print("\nGenerated files:")
print("  - 01_word_frequency.png")
print("  - 02_lda_topic_heatmaps.png")
print("  - topic_modeling_report.txt")
print("\n" + "=" * 80)
