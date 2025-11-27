"""
Phase 6: Perspective API Analysis
Critical analysis of discourse toxicity and harmful content using Google's Perspective API.
Attributes analyzed: TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, THREAT, INSULT, PROFANITY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient import discovery
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create output directory
os.makedirs('perspective_output', exist_ok=True)

print("="*80)
print("PHASE 6: PERSPECTIVE API CRITICAL ANALYSIS")
print("="*80)

# ============================================================================
# 1. SETUP API CLIENT
# ============================================================================
API_KEY = os.getenv('PERSPECTIVE_API_KEY')

if not API_KEY:
    raise ValueError("API Key not found. Please set PERSPECTIVE_API_KEY in .env file")

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

def get_perspective_scores(text):
    """
    Get scores for multiple attributes from Perspective API
    """
    if pd.isna(text) or text == '' or len(str(text)) < 2:
        return None
        
    analyze_request = {
        'comment': {'text': str(text)[:3000]}, # Limit length to avoid errors
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'THREAT': {},
            'PROFANITY': {}
        },
        'languages': ['en']
    }
    
    try:
        response = client.comments().analyze(body=analyze_request).execute()
        scores = {}
        for attr in response['attributeScores']:
            scores[attr] = response['attributeScores'][attr]['summaryScore']['value']
        return scores
    except Exception as e:
        # print(f"Error: {e}") # Suppress individual errors to keep output clean
        return None

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df = pd.read_csv('sentiment_output/reddit_with_sentiment.csv')
youtube_df = pd.read_csv('sentiment_output/youtube_with_sentiment.csv')

# Filter valid labels
valid_labels = ['P', 'I', 'N']
reddit_df = reddit_df[reddit_df['Label'].isin(valid_labels)]
youtube_df = youtube_df[youtube_df['label'].isin(valid_labels)]

# Sample data to stay within API quotas (Perspective has rate limits)
# Let's take a stratified sample of 500 from each platform for demonstration/analysis
# In a full run, you might want to run this in batches over time.
SAMPLE_SIZE = 300 

print(f"Sampling {SAMPLE_SIZE} comments per platform for API analysis...")
r_sample = reddit_df.groupby('Label', group_keys=False).apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE // 3)))
y_sample = youtube_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE // 3)))

print(f"✓ Reddit Sample: {len(r_sample)}")
print(f"✓ YouTube Sample: {len(y_sample)}")

# ============================================================================
# 3. RUN ANALYSIS
# ============================================================================
def process_batch(df, text_col):
    results = []
    texts = df[text_col].tolist()
    
    # Use ThreadPool for faster API calls (be careful with rate limits - 1 QPS usually)
    # We'll use a simple loop with sleep to be safe and respectful of the free tier
    
    print("   Processing comments (this takes time due to API rate limits)...")
    for text in tqdm(texts):
        scores = get_perspective_scores(text)
        if scores:
            results.append(scores)
            time.sleep(1.1) # Sleep to respect ~60 QPM limit
        else:
            results.append({
                'TOXICITY': np.nan, 'SEVERE_TOXICITY': np.nan, 
                'IDENTITY_ATTACK': np.nan, 'INSULT': np.nan, 
                'THREAT': np.nan, 'PROFANITY': np.nan
            })
            
    return pd.DataFrame(results)

print("\n🚀 Analyzing Reddit Data...")
r_scores = process_batch(r_sample, 'clean_text_comments')
r_sample = r_sample.reset_index(drop=True)
r_final = pd.concat([r_sample, r_scores], axis=1)

print("\n🚀 Analyzing YouTube Data...")
y_scores = process_batch(y_sample, 'text')
y_sample = y_sample.reset_index(drop=True)
y_final = pd.concat([y_sample, y_scores], axis=1)

# Save raw results
r_final.to_csv('perspective_output/reddit_perspective.csv', index=False)
y_final.to_csv('perspective_output/youtube_perspective.csv', index=False)

# ============================================================================
# 4. VISUALIZATION & ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("GENERATING CRITICAL INSIGHTS")
print("="*80)

attributes = ['TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'THREAT']
labels_map = {'P': 'Pro-Palestine', 'I': 'Pro-Israel', 'N': 'Neutral'}

# 4.1 Platform Comparison (Mean Scores)
print("\n1. Platform Toxicity Comparison")
r_means = r_final[attributes].mean()
y_means = y_final[attributes].mean()

comparison_df = pd.DataFrame({'Reddit': r_means, 'YouTube': y_means})
print(comparison_df)

fig, ax = plt.subplots(figsize=(10, 6))
comparison_df.plot(kind='bar', ax=ax, color=['#FF5722', '#FF0000'], alpha=0.8)
ax.set_title('Average Harmful Content Scores by Platform', fontsize=14, fontweight='bold')
ax.set_ylabel('Perspective API Score (0-1)')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('perspective_output/01_platform_toxicity_comparison.png', dpi=300)
print("✓ Saved: 01_platform_toxicity_comparison.png")

# 4.2 Toxicity by Stance (Reddit)
print("\n2. Reddit Toxicity by Stance")
r_final['Label_Full'] = r_final['Label'].map(labels_map)
r_stance_means = r_final.groupby('Label_Full')[attributes].mean()
print(r_stance_means)

fig, ax = plt.subplots(figsize=(12, 6))
r_stance_means.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Reddit: Harmful Content Attributes by Stance', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Score')
plt.legend(title='Attribute')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('perspective_output/02_reddit_toxicity_by_stance.png', dpi=300)
print("✓ Saved: 02_reddit_toxicity_by_stance.png")

# 4.3 Toxicity by Stance (YouTube)
print("\n3. YouTube Toxicity by Stance")
y_final['Label_Full'] = y_final['label'].map(labels_map)
y_stance_means = y_final.groupby('Label_Full')[attributes].mean()
print(y_stance_means)

fig, ax = plt.subplots(figsize=(12, 6))
y_stance_means.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('YouTube: Harmful Content Attributes by Stance', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Score')
plt.legend(title='Attribute')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('perspective_output/03_youtube_toxicity_by_stance.png', dpi=300)
print("✓ Saved: 03_youtube_toxicity_by_stance.png")

# 4.4 Identity Attack Distribution
print("\n4. Identity Attack Analysis")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(x='Label_Full', y='IDENTITY_ATTACK', data=r_final, ax=axes[0], palette='Set2')
axes[0].set_title('Reddit: Identity Attack Scores', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Identity Attack Score')

sns.boxplot(x='Label_Full', y='IDENTITY_ATTACK', data=y_final, ax=axes[1], palette='Set2')
axes[1].set_title('YouTube: Identity Attack Scores', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Identity Attack Score')

plt.tight_layout()
plt.savefig('perspective_output/04_identity_attack_distribution.png', dpi=300)
print("✓ Saved: 04_identity_attack_distribution.png")

# ============================================================================
# 5. GENERATE REPORT
# ============================================================================
with open('perspective_output/perspective_analysis_report.txt', 'w') as f:
    f.write("PERSPECTIVE API CRITICAL ANALYSIS REPORT\n")
    f.write("="*50 + "\n\n")
    
    f.write("1. PLATFORM COMPARISON (Mean Scores)\n")
    f.write("-" * 30 + "\n")
    f.write(comparison_df.to_string())
    f.write("\n\n")
    
    f.write("2. REDDIT BREAKDOWN BY STANCE\n")
    f.write("-" * 30 + "\n")
    f.write(r_stance_means.to_string())
    f.write("\n\n")
    
    f.write("3. YOUTUBE BREAKDOWN BY STANCE\n")
    f.write("-" * 30 + "\n")
    f.write(y_stance_means.to_string())
    f.write("\n\n")
    
    f.write("4. KEY INSIGHTS\n")
    f.write("-" * 30 + "\n")
    
    # Auto-generate insights based on data
    max_tox_r = r_stance_means['TOXICITY'].idxmax()
    max_tox_y = y_stance_means['TOXICITY'].idxmax()
    
    f.write(f"- Most Toxic Stance on Reddit: {max_tox_r} ({r_stance_means.loc[max_tox_r, 'TOXICITY']:.3f})\n")
    f.write(f"- Most Toxic Stance on YouTube: {max_tox_y} ({y_stance_means.loc[max_tox_y, 'TOXICITY']:.3f})\n")
    
    if r_means['IDENTITY_ATTACK'] > y_means['IDENTITY_ATTACK']:
        f.write("- Reddit has higher levels of Identity Attacks than YouTube.\n")
    else:
        f.write("- YouTube has higher levels of Identity Attacks than Reddit.\n")
        
    f.write("- 'Identity Attack' scores indicate discourse targeting race, religion, or ethnicity.\n")
    f.write("- 'Threat' scores indicate severe hostility or intent to harm.\n")

print("✓ Saved: perspective_analysis_report.txt")
print("\n" + "="*80)
print("✅ PERSPECTIVE ANALYSIS COMPLETE")
print("="*80)
