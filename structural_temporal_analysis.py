"""
Phase 4: Structural & Temporal Analysis
RQ2: Network Analysis (User-Post)
RQ3: Response Time & Thread Depth (Volume)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Output directory is already created by advanced_analysis.py, but good to be safe
os.makedirs('advanced_analysis_output', exist_ok=True)

print("="*80)
print("PHASE 4: STRUCTURAL & TEMPORAL ANALYSIS")
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
# 2. NETWORK ANALYSIS (RQ2) - REDDIT
# ============================================================================
print("\n" + "="*80)
print("RQ2: NETWORK ANALYSIS (USER ACTIVITY)")
print("="*80)

# Since we lack parent_id for full trees, we analyze User-Post activity
if 'author_name' in reddit_df.columns:
    user_counts = reddit_df['author_name'].value_counts()
    top_users = user_counts.head(20)
    
    print("\nTop 10 Most Active Users (Reddit):")
    print(top_users.head(10))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=top_users.values, y=top_users.index, ax=ax, palette='viridis')
    ax.set_title('Top 20 Most Active Users (Reddit)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Comments')
    plt.tight_layout()
    plt.savefig('advanced_analysis_output/06_network_top_users.png', dpi=300)
    print("✓ Saved: 06_network_top_users.png")
    
    # User Stance Consistency (if users comment multiple times)
    # Find users with > 5 comments
    active_users = user_counts[user_counts >= 5].index
    user_stance_df = reddit_df[reddit_df['author_name'].isin(active_users)]
    
    # Calculate dominant stance per user
    user_stance_dominance = user_stance_df.groupby('author_name')['Label'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    
    print("\nStance Distribution of Active Users (>5 comments):")
    print(user_stance_dominance.value_counts())

else:
    print("⚠️ Author name missing for network analysis.")

# ============================================================================
# 3. RESPONSE TIME ANALYSIS (RQ3) - REDDIT ONLY
# ============================================================================
print("\n" + "="*80)
print("RQ3: RESPONSE TIME ANALYSIS (IMMEDIACY)")
print("="*80)

if 'created_time' in reddit_df.columns and 'post_created_time' in reddit_df.columns:
    # Convert to datetime
    reddit_df['created_time'] = pd.to_datetime(reddit_df['created_time'], errors='coerce')
    reddit_df['post_created_time'] = pd.to_datetime(reddit_df['post_created_time'], errors='coerce')
    
    # Calculate difference in minutes
    reddit_df['response_time_mins'] = (reddit_df['created_time'] - reddit_df['post_created_time']).dt.total_seconds() / 60
    
    # Filter out negative times (data errors) or extremely long times (archived posts)
    # Let's look at first 48 hours (2880 mins)
    response_df = reddit_df[(reddit_df['response_time_mins'] > 0) & (reddit_df['response_time_mins'] < 2880)]
    
    print(f"Analyzed {len(response_df)} comments within 48 hours of posting.")
    print(f"Median Response Time: {response_df['response_time_mins'].median():.2f} minutes")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(response_df['response_time_mins'], bins=50, kde=True, color='#e74c3c', ax=ax)
    ax.set_title('Distribution of Response Times (Reddit)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time since Post Creation (Minutes)')
    ax.set_xlim(0, 1440) # Show first 24 hours
    plt.tight_layout()
    plt.savefig('advanced_analysis_output/07_response_time_distribution.png', dpi=300)
    print("✓ Saved: 07_response_time_distribution.png")
    
    # Response Time by Stance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Label', y='response_time_mins', data=response_df, ax=ax, palette='Set2', showfliers=False)
    ax.set_title('Response Time by Stance (First 48h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Response Time (Minutes)')
    plt.tight_layout()
    plt.savefig('advanced_analysis_output/08_response_time_by_stance.png', dpi=300)
    print("✓ Saved: 08_response_time_by_stance.png")

else:
    print("⚠️ Timestamp data missing for response time analysis.")

# ============================================================================
# 4. THREAD DEPTH / CONVERSATION VOLUME (RQ3)
# ============================================================================
print("\n" + "="*80)
print("RQ3: CONVERSATION VOLUME (PROXY FOR DEPTH)")
print("="*80)

# Reddit: Comments per Post
if 'post_id' in reddit_df.columns:
    reddit_vol = reddit_df.groupby('post_id').size()
    print(f"\nReddit: Avg Comments per Post (in dataset): {reddit_vol.mean():.2f}")
else:
    reddit_vol = pd.Series([])

# YouTube: Comments per Video
if 'video id' in youtube_df.columns:
    youtube_vol = youtube_df.groupby('video id').size()
    print(f"YouTube: Avg Comments per Video (in dataset): {youtube_vol.mean():.2f}")
else:
    youtube_vol = pd.Series([])

# Visualization
if len(reddit_vol) > 0 and len(youtube_vol) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize for comparison? Or just raw counts?
    # Raw counts might be misleading if sampling was different.
    # But let's show the distribution.
    
    sns.kdeplot(reddit_vol, label='Reddit (Comments/Post)', shade=True, color='#3498db', ax=ax)
    sns.kdeplot(youtube_vol, label='YouTube (Comments/Video)', shade=True, color='#e74c3c', ax=ax)
    
    ax.set_title('Conversation Volume Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Comments')
    ax.set_xlim(0, 100) # Limit x-axis for readability
    ax.legend()
    plt.tight_layout()
    plt.savefig('advanced_analysis_output/09_conversation_volume.png', dpi=300)
    print("✓ Saved: 09_conversation_volume.png")

print("\n" + "="*80)
print("✅ STRUCTURAL & TEMPORAL ANALYSIS COMPLETE")
print("="*80)
