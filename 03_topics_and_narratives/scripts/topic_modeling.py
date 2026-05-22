"""
Topic Modeling for Israel-Hamas War Discourse
RQ2: What distinct topics and narratives emerge on each platform?
Uses LDA, NMF, n-gram analysis, stance-specific topic models,
and TF-IDF distinctive term extraction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import re
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '03_topics_and_narratives' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_TEXT_COL   = 'self_text'
YOUTUBE_TEXT_COL  = 'text'
REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

print("=" * 80)
print("TOPIC MODELING - ISRAEL-HAMAS WAR DISCOURSE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading sentiment-enhanced data...")
reddit_df  = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')
youtube_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'youtube_with_sentiment.csv')

print(f"✔ Reddit:  {len(reddit_df):,} rows")
print(f"✔ YouTube: {len(youtube_df):,} rows")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
print("\n🔧 Preprocessing text data...")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

reddit_df['cleaned_text']  = reddit_df[REDDIT_TEXT_COL].apply(clean_text)
youtube_df['cleaned_text'] = youtube_df[YOUTUBE_TEXT_COL].apply(clean_text)

reddit_df  = reddit_df[reddit_df['cleaned_text'].str.len() > 20].copy()
youtube_df = youtube_df[youtube_df['cleaned_text'].str.len() > 20].copy()

print(f"✔ After cleaning — Reddit: {len(reddit_df):,} | YouTube: {len(youtube_df):,}")

# ============================================================================
# STOPWORDS
# ============================================================================
domain_stopwords = [
    'israel', 'israeli', 'israelis', 'palestine', 'palestinian', 'palestinians',
    'hamas', 'gaza', 'war', 'conflict', 'just', 'like', 'people', 'know',
    'think', 'going', 'said', 'really', 'also', 'would', 'could', 'one',
    'two', 'even', 'make', 'get', 'want', 'see', 'say', 'tell', 'much',
    'many', 'thing', 'way', 'time', 'come', 'went', 'yes', 'no', 'ok',
]

# ============================================================================
# HELPER: TOP WORDS
# ============================================================================
def get_top_words(texts, n=20, extra_stopwords=None):
    sw = 'english'
    if extra_stopwords:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        sw = list(ENGLISH_STOP_WORDS.union(set(extra_stopwords)))
    vec = CountVectorizer(max_features=n * 3, stop_words=sw)
    try:
        X = vec.fit_transform(texts)
        freq = dict(zip(vec.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
        return Counter(freq).most_common(n)
    except Exception:
        return []

# ============================================================================
# WORD FREQUENCY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("WORD FREQUENCY ANALYSIS")
print("=" * 80)

reddit_top_words  = get_top_words(reddit_df['cleaned_text'],  n=20, extra_stopwords=domain_stopwords)
youtube_top_words = get_top_words(youtube_df['cleaned_text'], n=20, extra_stopwords=domain_stopwords)

print("\n📊 TOP 20 WORDS – REDDIT:")
for word, count in reddit_top_words:
    print(f"  {word}: {count:,}")

print("\n📊 TOP 20 WORDS – YOUTUBE:")
for word, count in youtube_top_words:
    print(f"  {word}: {count:,}")

print("\n📊 TOP WORDS BY STANCE – REDDIT:")
for stance in ['P', 'I', 'N']:
    texts = reddit_df[reddit_df[REDDIT_LABEL_COL] == stance]['cleaned_text']
    if len(texts) > 0:
        print(f"\n  Stance {stance}:")
        for word, count in get_top_words(texts, n=10, extra_stopwords=domain_stopwords):
            print(f"    {word}: {count:,}")

print("\n📊 TOP WORDS BY STANCE – YOUTUBE:")
for stance in ['P', 'I', 'N']:
    texts = youtube_df[youtube_df[YOUTUBE_LABEL_COL] == stance]['cleaned_text']
    if len(texts) > 0:
        print(f"\n  Stance {stance}:")
        for word, count in get_top_words(texts, n=10, extra_stopwords=domain_stopwords):
            print(f"    {word}: {count:,}")

# ============================================================================
# N-GRAM ANALYSIS (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("N-GRAM ANALYSIS (Bigrams & Trigrams)")
print("=" * 80)

def get_ngrams(texts, ngram_range=(2, 2), n=15, extra_stopwords=None):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    sw = list(ENGLISH_STOP_WORDS.union(set(extra_stopwords or [])))
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=sw,
                          max_features=n * 5, min_df=3)
    try:
        X = vec.fit_transform(texts)
        freq = dict(zip(vec.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
        return Counter(freq).most_common(n)
    except Exception:
        return []

for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                        ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
    print(f"\n📊 {name} – Top Bigrams:")
    for bg, cnt in get_ngrams(df['cleaned_text'], (2,2), 10, domain_stopwords):
        print(f"  '{bg}': {cnt:,}")
    print(f"\n📊 {name} – Top Trigrams:")
    for tg, cnt in get_ngrams(df['cleaned_text'], (3,3), 10, domain_stopwords):
        print(f"  '{tg}': {cnt:,}")

# ============================================================================
# TF-IDF DISTINCTIVE TERMS PER STANCE (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("TF-IDF DISTINCTIVE TERMS PER STANCE (NEW)")
print("=" * 80)

def tfidf_top_terms_per_stance(df, text_col, label_col, n=10, extra_sw=None):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    sw = list(ENGLISH_STOP_WORDS.union(set(extra_sw or [])))
    results = {}
    for stance in ['P', 'I', 'N']:
        texts = df[df[label_col] == stance][text_col].fillna('').tolist()
        if len(texts) < 5:
            continue
        vec = TfidfVectorizer(max_features=5000, stop_words=sw, ngram_range=(1, 2))
        try:
            X = vec.fit_transform(texts)
            mean_tfidf = X.mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[-n:][::-1]
            terms = [(vec.get_feature_names_out()[i], mean_tfidf[i]) for i in top_idx]
            results[stance] = terms
        except Exception:
            pass
    return results

for name, df, tcol, lcol in [
        ("Reddit",  reddit_df,  'cleaned_text', REDDIT_LABEL_COL),
        ("YouTube", youtube_df, 'cleaned_text', YOUTUBE_LABEL_COL)]:
    print(f"\n📊 {name} – TF-IDF Distinctive Terms per Stance:")
    stance_terms = tfidf_top_terms_per_stance(df, tcol, lcol, n=10, extra_sw=domain_stopwords)
    for stance, terms in stance_terms.items():
        stance_name = {'P': 'Pro-Palestine', 'I': 'Pro-Israel', 'N': 'Neutral'}[stance]
        print(f"  {stance_name}: {', '.join([t[0] for t in terms])}")

# ============================================================================
# LDA TOPIC MODELING
# ============================================================================
print("\n" + "=" * 80)
print("LDA TOPIC MODELING")
print("=" * 80)

N_TOPICS = 5
N_TOP_WORDS = 10

def run_lda(texts, n_topics=N_TOPICS, max_features=1500):
    vec = CountVectorizer(max_df=0.92, min_df=5, max_features=max_features,
                          stop_words='english')
    tf = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                     max_iter=25, learning_method='online')
    lda.fit(tf)
    return lda, vec, tf

def print_topics(model, vectorizer, n_words=N_TOP_WORDS, platform_name=""):
    fn = vectorizer.get_feature_names_out()
    print(f"\n📊 {platform_name} – LDA TOPICS:")
    topics = {}
    for i, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-n_words:][::-1]
        words = [fn[j] for j in top_idx]
        topics[i] = words
        print(f"  Topic {i+1}: {', '.join(words)}")
    return topics

print("\n🔍 Running LDA on Reddit...")
reddit_lda,  reddit_lda_vec,  reddit_tf  = run_lda(reddit_df['cleaned_text'])
reddit_topics = print_topics(reddit_lda, reddit_lda_vec, platform_name="Reddit")
print("\n🔍 Running LDA on YouTube...")
youtube_lda, youtube_lda_vec, youtube_tf = run_lda(youtube_df['cleaned_text'])
youtube_topics = print_topics(youtube_lda, youtube_lda_vec, platform_name="YouTube")

# Assign dominant topic to each comment
def get_doc_topics(model, tf_matrix):
    doc_topic = model.transform(tf_matrix)
    return np.argmax(doc_topic, axis=1)

reddit_df['dominant_topic']  = get_doc_topics(reddit_lda,  reddit_tf)
youtube_df['dominant_topic'] = get_doc_topics(youtube_lda, youtube_tf)

# ============================================================================
# STANCE-SPECIFIC LDA (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("STANCE-SPECIFIC TOPIC MODELING (NEW)")
print("=" * 80)

for name, df, lcol in [("Reddit", reddit_df, REDDIT_LABEL_COL),
                        ("YouTube", youtube_df, YOUTUBE_LABEL_COL)]:
    for stance in ['P', 'I', 'N']:
        texts = df[df[lcol] == stance]['cleaned_text']
        if len(texts) < 50:
            continue
        print(f"\n📊 {name} – {stance} Stance:")
        try:
            lda_s, vec_s, _ = run_lda(texts, n_topics=3, max_features=800)
            fn = vec_s.get_feature_names_out()
            for i, topic in enumerate(lda_s.components_):
                top = topic.argsort()[-8:][::-1]
                print(f"  Topic {i+1}: {', '.join([fn[j] for j in top])}")
        except Exception as e:
            print(f"  Skipped ({e})")

# ============================================================================
# NMF TOPIC MODELING
# ============================================================================
print("\n" + "=" * 80)
print("NMF TOPIC MODELING")
print("=" * 80)

def run_nmf(texts, n_topics=N_TOPICS, max_features=1500):
    vec = TfidfVectorizer(max_df=0.92, min_df=5, max_features=max_features,
                          stop_words='english')
    tfidf = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
    nmf.fit(tfidf)
    return nmf, vec

print("\n🔍 Running NMF on Reddit...")
reddit_nmf,  reddit_nmf_vec  = run_nmf(reddit_df['cleaned_text'])
print_topics(reddit_nmf, reddit_nmf_vec, platform_name="Reddit NMF")

print("\n🔍 Running NMF on YouTube...")
youtube_nmf, youtube_nmf_vec = run_nmf(youtube_df['cleaned_text'])
print_topics(youtube_nmf, youtube_nmf_vec, platform_name="YouTube NMF")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# --- 01: Word frequency bar charts ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Top 20 Most Frequent Words by Platform', fontsize=15, fontweight='bold')

for ax, top_words, title, color in [
        (axes[0], reddit_top_words,  'Reddit',  '#3498db'),
        (axes[1], youtube_top_words, 'YouTube', '#e74c3c')]:
    words  = [w for w, _ in top_words]
    counts = [c for _, c in top_words]
    ax.barh(words[::-1], counts[::-1], color=color, alpha=0.85, edgecolor='black')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency'); ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_word_frequency.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 01_word_frequency.png")
plt.close()

# --- 02: LDA topic-word heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('LDA Topic-Word Distribution', fontsize=15, fontweight='bold')

for ax, model, vec, title in [
        (axes[0], reddit_lda,  reddit_lda_vec,  'Reddit'),
        (axes[1], youtube_lda, youtube_lda_vec, 'YouTube')]:
    fn = vec.get_feature_names_out()
    top_words_all = []
    heat_data = []
    for topic in model.components_:
        top_idx = topic.argsort()[-10:][::-1]
        top_words_all.append([fn[i] for i in top_idx])
        heat_data.append(topic[top_idx])
    sns.heatmap(heat_data, xticklabels=top_words_all[0],
                yticklabels=[f'Topic {i+1}' for i in range(N_TOPICS)],
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Weight'})
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Top Words'); ax.set_ylabel('Topics')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_lda_topic_heatmaps.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 02_lda_topic_heatmaps.png")
plt.close()

# --- 03: Dominant topic distribution by stance ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Dominant Topic Distribution by Stance', fontsize=15, fontweight='bold')

for ax, df, lcol, title in [
        (axes[0], reddit_df,  REDDIT_LABEL_COL,  'Reddit'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube')]:
    if lcol in df.columns:
        ct = pd.crosstab(df[lcol], df['dominant_topic'], normalize='index') * 100
        ct = ct.reindex(index=[c for c in ['P', 'I', 'N'] if c in ct.index])
        ct.columns = [f'T{i+1}' for i in ct.columns]
        ct.plot(kind='bar', ax=ax, stacked=True, cmap='tab10', edgecolor='black', alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Stance'); ax.set_ylabel('Percentage (%)')
        ax.legend(title='Topic', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_topic_distribution_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 03_topic_distribution_by_stance.png")
plt.close()

# --- 04: Bigram comparison ---
r_bigrams  = get_ngrams(reddit_df['cleaned_text'],  (2, 2), 15, domain_stopwords)
y_bigrams  = get_ngrams(youtube_df['cleaned_text'], (2, 2), 15, domain_stopwords)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Top 15 Bigrams by Platform', fontsize=15, fontweight='bold')

for ax, bigrams, title, color in [
        (axes[0], r_bigrams, 'Reddit',  '#3498db'),
        (axes[1], y_bigrams, 'YouTube', '#e74c3c')]:
    if bigrams:
        phrases = [b[0] for b in bigrams][::-1]
        counts  = [b[1] for b in bigrams][::-1]
        ax.barh(phrases, counts, color=color, alpha=0.85, edgecolor='black')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Frequency'); ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_bigram_analysis.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 04_bigram_analysis.png")
plt.close()

# --- 05: TF-IDF distinctive terms per stance (stacked barh) ---
for name, df, lcol, fname in [
        ("Reddit",  reddit_df,  REDDIT_LABEL_COL,  '05_reddit_tfidf_terms.png'),
        ("YouTube", youtube_df, YOUTUBE_LABEL_COL, '06_youtube_tfidf_terms.png')]:
    terms = tfidf_top_terms_per_stance(df, 'cleaned_text', lcol, n=12, extra_sw=domain_stopwords)
    n_stances = len(terms)
    if n_stances == 0:
        continue
    fig, axes_t = plt.subplots(1, n_stances, figsize=(7 * n_stances, 7))
    if n_stances == 1:
        axes_t = [axes_t]
    stance_colors = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}
    stance_names  = {'P': 'Pro-Palestine', 'I': 'Pro-Israel', 'N': 'Neutral'}
    for ax, (stance, term_list) in zip(axes_t, terms.items()):
        words  = [t[0] for t in term_list][::-1]
        scores = [t[1] for t in term_list][::-1]
        ax.barh(words, scores, color=stance_colors.get(stance, 'gray'),
                alpha=0.85, edgecolor='black')
        ax.set_title(f'{name}: {stance_names.get(stance, stance)}\nDistinctive Terms (TF-IDF)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean TF-IDF Score'); ax.grid(axis='x', alpha=0.3)
    plt.suptitle(f'{name} – TF-IDF Top Terms per Stance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches='tight')
    print(f"✔ Saved: {fname}")
    plt.close()

# ============================================================================
# SAVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SAVING TOPIC MODELING REPORT")
print("=" * 80)

with open(OUTPUT_DIR / 'topic_modeling_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\nTOPIC MODELING REPORT\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. REDDIT – TOP 20 WORDS\n" + "-" * 80 + "\n")
    for w, c in reddit_top_words:
        f.write(f"  {w}: {c:,}\n")

    f.write("\n2. YOUTUBE – TOP 20 WORDS\n" + "-" * 80 + "\n")
    for w, c in youtube_top_words:
        f.write(f"  {w}: {c:,}\n")

    f.write(f"\n3. REDDIT – LDA TOPICS\n" + "-" * 80 + "\n")
    fn = reddit_lda_vec.get_feature_names_out()
    for i, topic in enumerate(reddit_lda.components_):
        top = topic.argsort()[-N_TOP_WORDS:][::-1]
        f.write(f"  Topic {i+1}: {', '.join([fn[j] for j in top])}\n")

    f.write(f"\n4. YOUTUBE – LDA TOPICS\n" + "-" * 80 + "\n")
    fn_y = youtube_lda_vec.get_feature_names_out()
    for i, topic in enumerate(youtube_lda.components_):
        top = topic.argsort()[-N_TOP_WORDS:][::-1]
        f.write(f"  Topic {i+1}: {', '.join([fn_y[j] for j in top])}\n")

    f.write(f"\n5. REDDIT – NMF TOPICS\n" + "-" * 80 + "\n")
    fn_rn = reddit_nmf_vec.get_feature_names_out()
    for i, topic in enumerate(reddit_nmf.components_):
        top = topic.argsort()[-N_TOP_WORDS:][::-1]
        f.write(f"  Topic {i+1}: {', '.join([fn_rn[j] for j in top])}\n")

    f.write(f"\n6. YOUTUBE – NMF TOPICS\n" + "-" * 80 + "\n")
    fn_yn = youtube_nmf_vec.get_feature_names_out()
    for i, topic in enumerate(youtube_nmf.components_):
        top = topic.argsort()[-N_TOP_WORDS:][::-1]
        f.write(f"  Topic {i+1}: {', '.join([fn_yn[j] for j in top])}\n")

    f.write(f"\n7. REDDIT – TOP BIGRAMS\n" + "-" * 80 + "\n")
    for bg, cnt in r_bigrams:
        f.write(f"  '{bg}': {cnt:,}\n")

    f.write(f"\n8. YOUTUBE – TOP BIGRAMS\n" + "-" * 80 + "\n")
    for bg, cnt in y_bigrams:
        f.write(f"  '{bg}': {cnt:,}\n")

    f.write("\n" + "=" * 80 + "\n")

print("✔ Saved: topic_modeling_report.txt")
print("\n" + "=" * 80)
print("✅ TOPIC MODELING COMPLETE!")
print("=" * 80)
