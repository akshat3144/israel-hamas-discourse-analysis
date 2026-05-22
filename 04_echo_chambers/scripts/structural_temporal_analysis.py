"""
Structural & Temporal Analysis — RQ3
User activity, response time, conversation volume.
Column schema: Reddit uses author_name, created_time (Unix), post_created_time (Unix),
               post_id, Label; YouTube uses video id, created_time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '04_echo_chambers' / 'outputs' / 'advanced_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

palette = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}

print("=" * 80)
print("STRUCTURAL & TEMPORAL ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df  = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')
youtube_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'youtube_with_sentiment.csv')

print(f"✔ Reddit:  {len(reddit_df):,} rows")
print(f"✔ YouTube: {len(youtube_df):,} rows")

# ── Parse timestamps ──
# Reddit: Unix timestamp (seconds since epoch)
for col in ['created_time', 'post_created_time']:
    if col in reddit_df.columns:
        reddit_df[col] = pd.to_numeric(reddit_df[col], errors='coerce')
        reddit_df[col + '_dt'] = pd.to_datetime(reddit_df[col], unit='s', utc=True, errors='coerce')

# YouTube: ISO date strings
if 'created_time' in youtube_df.columns:
    youtube_df['created_time_dt'] = pd.to_datetime(youtube_df['created_time'], errors='coerce')
if 'video_date' in youtube_df.columns:
    youtube_df['video_date_dt'] = pd.to_datetime(youtube_df['video_date'], errors='coerce')

# ============================================================================
# 1. USER ACTIVITY (Reddit)
# ============================================================================
print("\n" + "=" * 80)
print("1. USER ACTIVITY ANALYSIS")
print("=" * 80)

if 'author_name' in reddit_df.columns:
    user_counts = reddit_df['author_name'].value_counts()
    top_users   = user_counts.head(20)

    print(f"\nTotal unique Reddit users: {len(user_counts):,}")
    print(f"Top 10 most active:\n{top_users.head(10)}")

    # 06: Top users bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=top_users.values, y=top_users.index, ax=ax, palette='viridis')
    ax.set_title('Top 20 Most Active Users (Reddit)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Comments'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '31_network_top_users.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 31_network_top_users.png")
    plt.close()

    # User stance consistency
    active_users = user_counts[user_counts >= 5].index
    if len(active_users) > 0:
        user_stance_df = reddit_df[reddit_df['author_name'].isin(active_users)]
        dom_stance = (user_stance_df.groupby('author_name')[REDDIT_LABEL_COL]
                      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'))
        print(f"\nActive Users (≥5 comments): {len(active_users):,}")
        print(f"Stance distribution:\n{dom_stance.value_counts()}")

if 'author' in youtube_df.columns:
    yt_user_counts = youtube_df['author'].value_counts()
    print(f"\nTotal unique YouTube commenters: {len(yt_user_counts):,}")
    print(f"Top 10:\n{yt_user_counts.head(10)}")

# ============================================================================
# 2. RESPONSE TIME (Reddit — Unix timestamps)
# ============================================================================
print("\n" + "=" * 80)
print("2. RESPONSE TIME ANALYSIS (Reddit)")
print("=" * 80)

if 'created_time' in reddit_df.columns and 'post_created_time' in reddit_df.columns:
    reddit_df['response_time_mins'] = (
        (reddit_df['created_time'] - reddit_df['post_created_time']) / 60)

    # Keep valid window: 0 to 7 days (10080 minutes)
    resp = reddit_df[
        (reddit_df['response_time_mins'] > 0) &
        (reddit_df['response_time_mins'] < 10080)
    ].copy()

    print(f"\nComments with valid response times: {len(resp):,}")
    print(f"Median response time: {resp['response_time_mins'].median():.1f} minutes "
          f"({resp['response_time_mins'].median()/60:.1f} hours)")
    print(f"Mean response time:   {resp['response_time_mins'].mean():.1f} minutes")

    # Response time stats by stance
    for s in ['P', 'I', 'N']:
        g = resp[resp[REDDIT_LABEL_COL] == s]['response_time_mins']
        if len(g) > 0:
            print(f"  Stance {s}: median={g.median():.1f} min, n={len(g):,}")

    # 07: Response time distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(resp['response_time_mins'].clip(upper=1440), bins=60,
                 color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[0].axvline(resp['response_time_mins'].median(), color='navy',
                    linestyle='--', label=f'Median: {resp["response_time_mins"].median():.0f} min')
    axes[0].set_title('Response Time Distribution (Reddit, first 24h)',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Minutes since post creation'); axes[0].set_ylabel('Frequency')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Response time by stance (boxplot)
    valid_stance = resp[resp[REDDIT_LABEL_COL].isin(['P', 'I', 'N'])]
    valid_stance['rt_clip'] = valid_stance['response_time_mins'].clip(upper=1440)
    sns.boxplot(x=REDDIT_LABEL_COL, y='rt_clip', data=valid_stance,
                palette=palette, ax=axes[1], showfliers=False, order=['P', 'I', 'N'])
    axes[1].set_title('Response Time by Stance (first 24h)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Stance'); axes[1].set_ylabel('Response Time (minutes)')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '32_response_time_analysis.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 32_response_time_analysis.png")
    plt.close()
else:
    print("⚠️  Timestamp columns not found — skipping response time.")

# ============================================================================
# 3. TEMPORAL TRENDS (month-over-month comment volume)
# ============================================================================
print("\n" + "=" * 80)
print("3. TEMPORAL ACTIVITY TRENDS")
print("=" * 80)

if 'created_time_dt' in reddit_df.columns:
    reddit_df['year_month'] = reddit_df['created_time_dt'].dt.to_period('M')
    r_temporal = reddit_df.groupby('year_month').size()
    print(f"\nReddit monthly volume:\n{r_temporal}")

if 'created_time_dt' in youtube_df.columns:
    youtube_df['year_month'] = youtube_df['created_time_dt'].dt.to_period('M')
    y_temporal = youtube_df.groupby('year_month').size()
    print(f"\nYouTube monthly volume:\n{y_temporal}")

# Combined temporal plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Monthly Comment Volume (Both Platforms)', fontsize=15, fontweight='bold')

for ax, df, label_col, title, color in [
        (axes[0], reddit_df, REDDIT_LABEL_COL, 'Reddit', '#3498db'),
        (axes[1], youtube_df, YOUTUBE_LABEL_COL, 'YouTube', '#e74c3c')]:
    if 'year_month' in df.columns:
        total = df.groupby('year_month').size()
        total.index = total.index.astype(str)
        ax.bar(total.index, total.values, color=color, alpha=0.7, edgecolor='black', label='Total')

        if label_col in df.columns:
            for s in ['P', 'I', 'N']:
                s_data = df[df[label_col] == s].groupby('year_month').size()
                s_data.index = s_data.index.astype(str)
                ax.plot(s_data.index, s_data.values, marker='o',
                        label=f'Stance {s}', color=palette.get(s), linewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Month'); ax.set_ylabel('Comment Count')
        ax.legend(title='Series'); ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '33_temporal_activity.png', dpi=300, bbox_inches='tight')
print("✔ Saved: 33_temporal_activity.png")
plt.close()

# ============================================================================
# 4. CONVERSATION VOLUME (comments per thread / video)
# ============================================================================
print("\n" + "=" * 80)
print("4. CONVERSATION VOLUME")
print("=" * 80)

reddit_vol  = pd.Series(dtype=float)
youtube_vol = pd.Series(dtype=float)

if 'post_id' in reddit_df.columns:
    reddit_vol = reddit_df.groupby('post_id').size()
    print(f"Reddit:  {len(reddit_vol):,} threads, avg {reddit_vol.mean():.1f} comments/thread")

if 'video id' in youtube_df.columns:
    youtube_vol = youtube_df.groupby('video id').size()
    print(f"YouTube: {len(youtube_vol):,} videos, avg {youtube_vol.mean():.1f} comments/video")

if len(reddit_vol) > 0 and len(youtube_vol) > 0:
    # Mann-Whitney U for distribution comparison
    u, p = mannwhitneyu(reddit_vol, youtube_vol, alternative='two-sided')
    print(f"\nConversation depth difference: U={u:.0f}, p={p:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        sns.kdeplot(reddit_vol.clip(upper=200),  label='Reddit (comments/post)',
                    fill=True, color='#3498db', ax=ax, alpha=0.4)
        sns.kdeplot(youtube_vol.clip(upper=200), label='YouTube (comments/video)',
                    fill=True, color='#e74c3c', ax=ax, alpha=0.4)
    except TypeError:
        # Older seaborn uses shade=
        sns.kdeplot(reddit_vol.clip(upper=200),  label='Reddit (comments/post)',
                    shade=True, color='#3498db', ax=ax, alpha=0.4)
        sns.kdeplot(youtube_vol.clip(upper=200), label='YouTube (comments/video)',
                    shade=True, color='#e74c3c', ax=ax, alpha=0.4)

    ax.set_title('Conversation Volume Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Comments (clipped at 200)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '34_conversation_volume.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 34_conversation_volume.png")
    plt.close()

print("\n" + "=" * 80)
print("✅ STRUCTURAL & TEMPORAL ANALYSIS COMPLETE")
print("=" * 80)
