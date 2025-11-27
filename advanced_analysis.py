"""
Phase 4: Advanced Analysis
RQ2: Regression Models & Algorithmic Amplification
RQ3: Narrative Complexity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import textstat
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('advanced_analysis_output', exist_ok=True)

print("="*80)
print("PHASE 4: ADVANCED ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df = pd.read_csv('sentiment_output/reddit_with_sentiment.csv')
youtube_df = pd.read_csv('sentiment_output/youtube_with_sentiment.csv')

print(f"✓ Reddit: {len(reddit_df)} rows")
print(f"✓ YouTube: {len(youtube_df)} rows")

# ============================================================================
# 2. REGRESSION MODELS (RQ2) - REDDIT ONLY
# ============================================================================
print("\n" + "="*80)
print("RQ2: REGRESSION MODELS (PREDICTING ENGAGEMENT)")
print("="*80)

# Prepare Reddit Data
# We want to predict 'score' based on 'vader_compound' (Sentiment) and 'Label' (Stance)
reg_df = reddit_df[['score', 'vader_compound', 'Label']].dropna()
reg_df['score'] = pd.to_numeric(reg_df['score'], errors='coerce')
reg_df = reg_df.dropna()

print(f"Data points for regression: {len(reg_df)}")

if len(reg_df) > 0:
    # Define model: Score ~ Sentiment + Stance
    # We treat 'Label' as categorical
    model = ols('score ~ vader_compound + C(Label)', data=reg_df).fit()
    
    print("\n--- Regression Results (Score ~ Sentiment + Stance) ---")
    print(model.summary())
    
    # Save results
    with open('advanced_analysis_output/regression_results.txt', 'w') as f:
        f.write(str(model.summary()))
    print("✓ Saved: regression_results.txt")
    
    # Visualization of coefficients
    params = model.params.drop('Intercept')
    conf = model.conf_int().drop('Intercept')
    conf.columns = ['Lower', 'Upper']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    params.plot(kind='barh', xerr=(params - conf['Lower']), ax=ax, color='#3498db')
    ax.set_title('Impact of Stance and Sentiment on Reddit Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Coefficient Value (Change in Score)')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('advanced_analysis_output/01_regression_coefficients.png', dpi=300)
    print("✓ Saved: 01_regression_coefficients.png")

else:
    print("⚠️ Not enough data for regression analysis.")

# ============================================================================
# 3. ALGORITHMIC AMPLIFICATION (RQ2) - REDDIT ONLY
# ============================================================================
print("\n" + "="*80)
print("RQ2: ALGORITHMIC AMPLIFICATION (CONTROVERSIALITY)")
print("="*80)

if 'controversiality' in reddit_df.columns and 'score' in reddit_df.columns:
    # Compare scores for controversial vs non-controversial
    # controversiality is usually 0 or 1
    
    amplification_df = reddit_df[['score', 'controversiality', 'Label']].dropna()
    amplification_df['controversiality'] = amplification_df['controversiality'].astype(int)
    
    # Calculate mean score by controversiality
    mean_scores = amplification_df.groupby('controversiality')['score'].mean()
    print("\nMean Score by Controversiality:")
    print(mean_scores)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='controversiality', y='score', data=amplification_df, ax=ax, palette='viridis')
    ax.set_title('Impact of Controversiality on Engagement (Reddit)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Controversial (0=No, 1=Yes)')
    ax.set_ylabel('Average Score')
    ax.set_xticklabels(['Non-Controversial', 'Controversial'])
    plt.tight_layout()
    plt.savefig('advanced_analysis_output/02_amplification_controversy.png', dpi=300)
    print("✓ Saved: 02_amplification_controversy.png")
    
    # Interaction with Stance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Label', y='score', hue='controversiality', data=amplification_df, ax=ax, palette='viridis')
    ax.set_title('Controversiality Amplification by Stance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stance')
    ax.set_ylabel('Average Score')
    plt.legend(title='Controversial', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig('advanced_analysis_output/03_amplification_by_stance.png', dpi=300)
    print("✓ Saved: 03_amplification_by_stance.png")

else:
    print("⚠️ Controversiality or Score data missing.")

# ============================================================================
# 4. NARRATIVE COMPLEXITY (RQ3)
# ============================================================================
print("\n" + "="*80)
print("RQ3: NARRATIVE COMPLEXITY (READABILITY & COHERENCE)")
print("="*80)

def calculate_complexity(text):
    """Calculate readability score (Flesch Reading Ease)"""
    if pd.isna(text) or text == '':
        return np.nan
    try:
        return textstat.flesch_reading_ease(str(text))
    except:
        return np.nan

print("Calculating readability scores (this may take a moment)...")
reddit_df['readability'] = reddit_df['self_text'].apply(calculate_complexity)
youtube_df['readability'] = youtube_df['text'].apply(calculate_complexity)

# Filter out invalid scores (textstat can return negative for gibberish)
reddit_df = reddit_df[reddit_df['readability'] > -100]
youtube_df = youtube_df[youtube_df['readability'] > -100]

print(f"Reddit Mean Readability: {reddit_df['readability'].mean():.2f}")
print(f"YouTube Mean Readability: {youtube_df['readability'].mean():.2f}")
print("(Higher score = Easier to read, Lower score = More complex)")

# Visualization: Platform Comparison
fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = [reddit_df['readability'].dropna(), youtube_df['readability'].dropna()]
ax.boxplot(data_to_plot, labels=['Reddit', 'YouTube'])
ax.set_title('Narrative Complexity: Readability Scores', fontsize=14, fontweight='bold')
ax.set_ylabel('Flesch Reading Ease Score')
plt.tight_layout()
plt.savefig('advanced_analysis_output/04_complexity_platform_comparison.png', dpi=300)
print("✓ Saved: 04_complexity_platform_comparison.png")

# Visualization: By Stance (Reddit)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(x='Label', y='readability', data=reddit_df, ax=axes[0], palette='Set2')
axes[0].set_title('Reddit Complexity by Stance', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Readability Score')

# Visualization: By Stance (YouTube)
label_col_yt = 'label' if 'label' in youtube_df.columns else 'Label'
sns.boxplot(x=label_col_yt, y='readability', data=youtube_df, ax=axes[1], palette='Set2')
axes[1].set_title('YouTube Complexity by Stance', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Readability Score')

plt.tight_layout()
plt.savefig('advanced_analysis_output/05_complexity_by_stance.png', dpi=300)
print("✓ Saved: 05_complexity_by_stance.png")

print("\n" + "="*80)
print("✅ ADVANCED ANALYSIS COMPLETE")
print("="*80)
