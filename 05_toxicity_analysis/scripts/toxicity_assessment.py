"""
Perspective API Toxicity Analysis — RQ4
Analyzes toxic speech patterns across platforms and stances.
Column schema: Reddit uses self_text + Label; YouTube uses text + Label.
Paths fixed: reads from 02_emotional_tone_analysis/outputs,
             writes to  05_toxicity_analysis/outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient import discovery
import json
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '05_toxicity_analysis' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

REDDIT_TEXT_COL   = 'self_text'
YOUTUBE_TEXT_COL  = 'text'
REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

print("=" * 80)
print("PERSPECTIVE API TOXICITY ANALYSIS — RQ4")
print("=" * 80)

# ============================================================================
# API SETUP
# ============================================================================
API_KEY = os.getenv('PERSPECTIVE_API_KEY')
if not API_KEY:
    raise ValueError("❌ PERSPECTIVE_API_KEY not found in .env file")

client = discovery.build(
    "commentanalyzer", "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl=(
        "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"),
    static_discovery=False,
)

ATTRIBUTES = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK',
              'INSULT', 'THREAT', 'PROFANITY']

def get_perspective_scores(text):
    if pd.isna(text) or len(str(text).strip()) < 5:
        return None
    analyze_request = {
        'comment': {'text': str(text)[:3000]},
        'requestedAttributes': {attr: {} for attr in ATTRIBUTES},
        'languages': ['en'],
    }
    try:
        resp = client.comments().analyze(body=analyze_request).execute()
        return {attr: resp['attributeScores'][attr]['summaryScore']['value']
                for attr in ATTRIBUTES}
    except Exception:
        return None

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df  = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')
youtube_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'youtube_with_sentiment.csv')

valid_labels = ['P', 'I', 'N']
reddit_df  = reddit_df[reddit_df[REDDIT_LABEL_COL].isin(valid_labels)].copy()
youtube_df = youtube_df[youtube_df[YOUTUBE_LABEL_COL].isin(valid_labels)].copy()

SAMPLE_SIZE = 300   # per platform (100 per stance)
print(f"Sampling {SAMPLE_SIZE} comments per platform for API analysis...")

r_sample = (reddit_df
            .groupby(REDDIT_LABEL_COL, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE // 3), random_state=42)))
y_sample = (youtube_df
            .groupby(YOUTUBE_LABEL_COL, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE // 3), random_state=42)))

print(f"✔ Reddit sample:  {len(r_sample):,}")
print(f"✔ YouTube sample: {len(y_sample):,}")

# ============================================================================
# PROCESS API CALLS
# ============================================================================
def process_batch(df, text_col, desc=""):
    results = []
    for text in tqdm(df[text_col].tolist(), desc=desc):
        scores = get_perspective_scores(text)
        if scores:
            results.append(scores)
            time.sleep(1.1)   # Respect ~60 QPM free-tier limit
        else:
            results.append({attr: np.nan for attr in ATTRIBUTES})
    return pd.DataFrame(results)

print("\n🚀 Analyzing Reddit comments...")
r_scores = process_batch(r_sample, REDDIT_TEXT_COL, desc="Reddit")
r_sample = r_sample.reset_index(drop=True)
r_final  = pd.concat([r_sample, r_scores], axis=1)

print("\n🚀 Analyzing YouTube comments...")
y_scores = process_batch(y_sample, YOUTUBE_TEXT_COL, desc="YouTube")
y_sample = y_sample.reset_index(drop=True)
y_final  = pd.concat([y_sample, y_scores], axis=1)

# Save raw results
r_final.to_csv(OUTPUT_DIR / 'reddit_perspective.csv',  index=False)
y_final.to_csv(OUTPUT_DIR / 'youtube_perspective.csv', index=False)
print("\n✔ Saved: reddit_perspective.csv")
print("✔ Saved: youtube_perspective.csv")

# ============================================================================
# ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS & VISUALIZATIONS")
print("=" * 80)

labels_map = {'P': 'Pro-Palestine', 'I': 'Pro-Israel', 'N': 'Neutral'}
r_final['Label_Full'] = r_final[REDDIT_LABEL_COL].map(labels_map)
y_final['Label_Full'] = y_final[YOUTUBE_LABEL_COL].map(labels_map)

CORE_ATTRS = ['TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'THREAT']

# ── Platform Means ──
r_means = r_final[CORE_ATTRS].mean()
y_means = y_final[CORE_ATTRS].mean()
comparison_df = pd.DataFrame({'Reddit': r_means, 'YouTube': y_means})
print(f"\nPlatform Toxicity Comparison:\n{comparison_df.round(4)}")

# Statistical tests: Reddit vs YouTube per attribute
print("\nMann-Whitney U Tests (Reddit vs YouTube):")
mwu_results = {}
for attr in CORE_ATTRS:
    g1 = r_final[attr].dropna()
    g2 = y_final[attr].dropna()
    if len(g1) > 1 and len(g2) > 1:
        u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
        mwu_results[attr] = {'U': u_stat, 'p': p_val, 'sig': p_val < 0.05}
        print(f"  {attr}: U={u_stat:.0f}, p={p_val:.6f} {'*' if p_val < 0.05 else ''}")

# ── Stance Means ──
r_stance = r_final.groupby('Label_Full')[CORE_ATTRS].mean()
y_stance = y_final.groupby('Label_Full')[CORE_ATTRS].mean()
print(f"\nReddit Toxicity by Stance:\n{r_stance.round(4)}")
print(f"\nYouTube Toxicity by Stance:\n{y_stance.round(4)}")

# Kruskal-Wallis across stances
print("\nKruskal-Wallis Tests (Stance differences):")
for name, df in [("Reddit", r_final), ("YouTube", y_final)]:
    for attr in CORE_ATTRS:
        groups = [df[df['Label_Full'] == s][attr].dropna()
                  for s in ['Pro-Palestine', 'Pro-Israel', 'Neutral']]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            h, p = kruskal(*groups)
            print(f"  {name} – {attr}: H={h:.3f}, p={p:.6f} {'*' if p < 0.05 else ''}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# --- 01: Platform toxicity comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
comparison_df.plot(kind='bar', ax=ax, color=['#FF5722', '#FF0000'], alpha=0.85, edgecolor='black')
ax.set_title('Average Harmful Content Scores by Platform',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Perspective API Score (0–1)')
ax.grid(axis='y', alpha=0.3); plt.xticks(rotation=0)
# Add significance stars
for i, attr in enumerate(CORE_ATTRS):
    if attr in mwu_results and mwu_results[attr]['sig']:
        ax.text(i, max(r_means[attr], y_means[attr]) + 0.005, '*',
                ha='center', fontsize=14, color='black')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_platform_toxicity_comparison.png', dpi=300, bbox_inches='tight')
print("\n✔ Saved: 01_platform_toxicity_comparison.png")
plt.close()

# --- 02: Reddit toxicity by stance ---
fig, ax = plt.subplots(figsize=(12, 6))
r_stance.plot(kind='bar', ax=ax, width=0.8, alpha=0.85, edgecolor='black')
ax.set_title('Reddit: Harmful Content Attributes by Stance',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Average Score'); plt.legend(title='Attribute'); plt.xticks(rotation=0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_reddit_toxicity_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 02_reddit_toxicity_by_stance.png")
plt.close()

# --- 03: YouTube toxicity by stance ---
fig, ax = plt.subplots(figsize=(12, 6))
y_stance.plot(kind='bar', ax=ax, width=0.8, alpha=0.85, edgecolor='black')
ax.set_title('YouTube: Harmful Content Attributes by Stance',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Average Score'); plt.legend(title='Attribute'); plt.xticks(rotation=0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_youtube_toxicity_by_stance.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 03_youtube_toxicity_by_stance.png")
plt.close()

# --- 04: Identity attack box plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df, title in [
        (axes[0], r_final, 'Reddit: Identity Attack Score by Stance'),
        (axes[1], y_final, 'YouTube: Identity Attack Score by Stance')]:
    sns.boxplot(x='Label_Full', y='IDENTITY_ATTACK', data=df, ax=ax,
                palette={'Pro-Palestine': '#2ecc71', 'Pro-Israel': '#3498db', 'Neutral': '#95a5a6'},
                order=['Pro-Palestine', 'Pro-Israel', 'Neutral'], showfliers=False)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Identity Attack Score'); ax.set_xlabel('')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_identity_attack_distribution.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 04_identity_attack_distribution.png")
plt.close()

# --- 05: Toxicity × Sentiment correlation (NEW) ---
for df_tox, name, color in [(r_final, 'Reddit', '#3498db'), (y_final, 'YouTube', '#e74c3c')]:
    if 'vader_compound' in df_tox.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, attr in zip(axes, ['TOXICITY', 'IDENTITY_ATTACK']):
            valid = df_tox[['vader_compound', attr]].dropna()
            if len(valid) > 5:
                ax.scatter(valid['vader_compound'], valid[attr],
                           alpha=0.5, s=25, color=color)
                from scipy.stats import spearmanr
                r, p = spearmanr(valid['vader_compound'], valid[attr])
                ax.set_title(f'{name}: Compound vs {attr}\nSpearman ρ={r:.3f}, p={p:.4f}',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('VADER Compound'); ax.set_ylabel(attr)
                ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f'05_{name.lower()}_toxicity_vs_sentiment.png'
        plt.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches='tight')
        print(f"✔ Saved: {fname}")
        plt.close()

# --- 06: Subreddit toxicity (NEW, Reddit only) ---
if 'subreddit' in r_final.columns:
    top_subs = r_final['subreddit'].value_counts().head(10).index
    sub_tox  = (r_final[r_final['subreddit'].isin(top_subs)]
                .groupby('subreddit')['TOXICITY']
                .agg(['mean', 'count'])
                .sort_values('mean', ascending=False))
    print(f"\nToxicity by Subreddit (top 10):\n{sub_tox.round(4)}")

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_c = ['#e74c3c' if v > 0.3 else '#27ae60' for v in sub_tox['mean']]
    ax.barh(sub_tox.index[::-1], sub_tox['mean'][::-1],
            color=bar_c[::-1], alpha=0.85, edgecolor='black')
    ax.axvline(0.3, color='gray', linestyle='--', alpha=0.6, label='Threshold (0.3)')
    ax.set_title('Mean Toxicity Score by Subreddit', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean TOXICITY Score'); ax.legend(); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_toxicity_by_subreddit.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 06_toxicity_by_subreddit.png")
    plt.close()

# ============================================================================
# REPORT
# ============================================================================
with open(OUTPUT_DIR / 'perspective_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("PERSPECTIVE API CRITICAL ANALYSIS REPORT\n" + "=" * 60 + "\n\n")

    f.write("1. PLATFORM COMPARISON (Mean Scores)\n" + "-" * 40 + "\n")
    f.write(comparison_df.round(4).to_string() + "\n\n")

    f.write("2. MANN-WHITNEY U TESTS (Platform Differences)\n" + "-" * 40 + "\n")
    for attr, res in mwu_results.items():
        f.write(f"  {attr}: U={res['U']:.0f}, p={res['p']:.6f} "
                f"{'SIGNIFICANT' if res['sig'] else 'not significant'}\n")
    f.write("\n")

    f.write("3. REDDIT BREAKDOWN BY STANCE\n" + "-" * 40 + "\n")
    f.write(r_stance.round(4).to_string() + "\n\n")

    f.write("4. YOUTUBE BREAKDOWN BY STANCE\n" + "-" * 40 + "\n")
    f.write(y_stance.round(4).to_string() + "\n\n")

    f.write("5. KEY INSIGHTS\n" + "-" * 40 + "\n")
    max_r = r_stance['TOXICITY'].idxmax() if not r_stance.empty else 'N/A'
    max_y = y_stance['TOXICITY'].idxmax() if not y_stance.empty else 'N/A'
    f.write(f"- Most Toxic Stance (Reddit):  {max_r}\n")
    f.write(f"- Most Toxic Stance (YouTube): {max_y}\n")
    f.write(f"- Reddit mean TOXICITY: {r_means['TOXICITY']:.4f}\n")
    f.write(f"- YouTube mean TOXICITY: {y_means['TOXICITY']:.4f}\n")
    higher_plat = 'Reddit' if r_means['IDENTITY_ATTACK'] > y_means['IDENTITY_ATTACK'] else 'YouTube'
    f.write(f"- Higher Identity Attack: {higher_plat}\n")

print("\n✔ Saved: perspective_analysis_report.txt")
print("\n" + "=" * 80)
print("✅ TOXICITY ANALYSIS COMPLETE")
print("=" * 80)
