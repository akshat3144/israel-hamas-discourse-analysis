"""
Network Analysis & Echo Chamber Detection — RQ3
1. User stance profiling
2. Homophily index (echo chamber detection)
3. Network visualization
4. Subreddit-level polarization (NEW)
5. User stance switching rate (NEW)
6. Community detection metrics (NEW)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '04_echo_chambers' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_LABEL_COL = 'Label'
palette = {'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'}

print("=" * 80)
print("NETWORK ANALYSIS & ECHO CHAMBER DETECTION")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')

valid_labels = ['P', 'I', 'N']
reddit_df = reddit_df[reddit_df[REDDIT_LABEL_COL].isin(valid_labels)].copy()
print(f"✔ Reddit (valid labels): {len(reddit_df):,} rows")

# ============================================================================
# 1. USER STANCE PROFILING
# ============================================================================
print("\n" + "-" * 60)
print("USER STANCE PROFILING")
print("-" * 60)

user_profiles = {}
active_users  = pd.Index([])

if 'author_name' in reddit_df.columns:
    user_counts  = reddit_df['author_name'].value_counts()
    active_users = user_counts[user_counts >= 3].index

    print(f"Total unique users:      {len(user_counts):,}")
    print(f"Active users (≥3 posts): {len(active_users):,}")

    for user in active_users:
        comments = reddit_df[reddit_df['author_name'] == user]
        stance_counts   = comments[REDDIT_LABEL_COL].value_counts()
        dominant_stance = stance_counts.idxmax()
        consistency     = stance_counts.max() / len(comments)
        n_stances_used  = (stance_counts > 0).sum()

        user_profiles[user] = {
            'dominant_stance': dominant_stance,
            'consistency':     consistency,
            'total_comments':  len(comments),
            'n_stances':       n_stances_used,
        }

    profiles_df = pd.DataFrame.from_dict(user_profiles, orient='index')

    print(f"\nDominant Stance Distribution of Active Users:")
    print(profiles_df['dominant_stance'].value_counts())
    print(f"\nMean Consistency: {profiles_df['consistency'].mean():.4f}")

    # ── 01: User stance distribution ──
    plt.figure(figsize=(8, 6))
    stance_counts_plot = profiles_df['dominant_stance'].value_counts().reindex(['P', 'I', 'N'])
    bar_colors = [palette.get(s, '#95a5a6') for s in stance_counts_plot.index]
    plt.bar(stance_counts_plot.index, stance_counts_plot.values,
            color=bar_colors, alpha=0.85, edgecolor='black')
    plt.title('Dominant Stance of Active Users (Reddit)', fontsize=14, fontweight='bold')
    plt.xlabel('Stance'); plt.ylabel('Number of Users'); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '21_user_stance_distribution.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 01_user_stance_distribution.png")
    plt.close()

    # ── 02: User consistency histogram ──
    plt.figure(figsize=(10, 6))
    sns.histplot(data=profiles_df, x='consistency', hue='dominant_stance',
                 element='step', palette=palette, bins=20, stat='count')
    plt.title('User Stance Consistency (1.0 = always same stance)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Consistency Score'); plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '22_user_consistency.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 02_user_consistency.png")
    plt.close()
else:
    print("⚠️  author_name column missing.")

# ============================================================================
# 2. USER STANCE SWITCHING RATE (NEW)
# ============================================================================
print("\n" + "-" * 60)
print("USER STANCE SWITCHING RATE (NEW)")
print("-" * 60)

if len(user_profiles) > 0:
    profiles_df_full = pd.DataFrame.from_dict(user_profiles, orient='index')
    switchers = (profiles_df_full['n_stances'] > 1).sum()
    total_active = len(profiles_df_full)
    switch_rate = switchers / total_active * 100
    print(f"Users who posted in >1 stance: {switchers:,} / {total_active:,} ({switch_rate:.1f}%)")

    # Consistency by dominant stance (table)
    cons_table = profiles_df_full.groupby('dominant_stance')['consistency'].agg(['mean', 'median', 'std'])
    print(f"\nConsistency by Dominant Stance:\n{cons_table.round(4)}")

# ============================================================================
# 3. ECHO CHAMBER DETECTION (HOMOPHILY INDEX)
# ============================================================================
print("\n" + "-" * 60)
print("ECHO CHAMBER DETECTION (HOMOPHILY INDEX)")
print("-" * 60)

homophily_data = []

if len(active_users) > 0 and 'post_id' in reddit_df.columns:
    # Thread dominant stance
    thread_stance = (reddit_df.groupby('post_id')[REDDIT_LABEL_COL]
                     .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'N'))

    for user in active_users:
        dom = user_profiles[user]['dominant_stance']
        comments = reddit_df[reddit_df['author_name'] == user]
        threads  = comments['post_id'].unique()

        matching = sum(1 for t in threads if thread_stance.get(t, None) == dom)
        total    = len(threads)
        if total > 0:
            homophily_data.append({
                'user':            user,
                'dominant_stance': dom,
                'homophily_index': matching / total,
                'total_threads':   total,
            })

    homophily_df = pd.DataFrame(homophily_data)
    print(f"\nUsers analysed: {len(homophily_df):,}")
    print(f"Overall mean homophily: {homophily_df['homophily_index'].mean():.4f}")
    print(f"\nHomophily by stance:")
    print(homophily_df.groupby('dominant_stance')['homophily_index']
          .agg(['mean', 'median', 'count']).round(4))

    echo_users = homophily_df[homophily_df['homophily_index'] > 0.8]
    print(f"\nEcho chamber users (>80% same-stance): "
          f"{len(echo_users):,} ({len(echo_users)/len(homophily_df)*100:.1f}%)")

    # ── 03: Homophily by stance box ──
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dominant_stance', y='homophily_index', data=homophily_df,
                palette=palette, order=['P', 'I', 'N'], showfliers=False)
    plt.title('Echo Chamber Effect: Homophily Index by Stance',
              fontsize=14, fontweight='bold')
    plt.ylabel('Homophily Index (% same-stance thread interaction)')
    plt.xlabel('User Dominant Stance'); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '23_echo_chamber_index.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 03_echo_chamber_index.png")
    plt.close()
else:
    print("⚠️  Cannot compute homophily (missing post_id or active users).")
    homophily_df = pd.DataFrame()

# ============================================================================
# 4. SUBREDDIT-LEVEL POLARIZATION (NEW)
# ============================================================================
print("\n" + "-" * 60)
print("SUBREDDIT-LEVEL POLARIZATION (NEW)")
print("-" * 60)

if 'subreddit' in reddit_df.columns:
    # Herfindahl-Hirschman Index (HHI) as stance concentration measure
    def herfindahl(series):
        counts = series.value_counts(normalize=True)
        return (counts ** 2).sum()

    top_subs = reddit_df['subreddit'].value_counts().head(15).index
    sub_data = reddit_df[reddit_df['subreddit'].isin(top_subs)]
    sub_polar = (sub_data.groupby('subreddit')[REDDIT_LABEL_COL]
                 .agg(['count', herfindahl])
                 .rename(columns={'count': 'n_comments', 'herfindahl': 'hhi'})
                 .sort_values('hhi', ascending=False))
    print(f"\nSubreddit Polarization (HHI — higher = more homogeneous):\n{sub_polar.round(4)}")

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = ['#e74c3c' if v > 0.5 else '#27ae60' for v in sub_polar['hhi']]
    ax.barh(sub_polar.index[::-1], sub_polar['hhi'][::-1],
            color=bar_colors[::-1], alpha=0.85, edgecolor='black')
    ax.axvline(0.33, color='gray', linestyle='--', alpha=0.6, label='Equal mix (HHI=0.33)')
    ax.set_title('Subreddit Stance Concentration (Herfindahl Index)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('HHI (0.33 = equal | 1.0 = homogeneous)')
    ax.legend(); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '25_subreddit_polarization.png', dpi=300, bbox_inches='tight')
    print("✔ Saved: 05_subreddit_polarization.png")
    plt.close()

# ============================================================================
# 5. NETWORK VISUALIZATION (Bipartite → User-User Projection)
# ============================================================================
print("\n" + "-" * 60)
print("NETWORK VISUALIZATION")
print("-" * 60)

if (len(active_users) > 0 and 'post_id' in reddit_df.columns
        and len(user_profiles) > 0):
    top_n = min(60, len(user_counts))
    top_users_graph = user_counts.head(top_n).index
    subset_df = reddit_df[reddit_df['author_name'].isin(top_users_graph)]

    B = nx.Graph()
    B.add_nodes_from(subset_df['author_name'].unique(), bipartite=0, ntype='user')
    B.add_nodes_from(subset_df['post_id'].unique(), bipartite=1, ntype='post')
    B.add_edges_from(zip(subset_df['author_name'], subset_df['post_id']))

    user_nodes = {n for n, d in B.nodes(data=True) if d.get('ntype') == 'user'}
    G = nx.bipartite.weighted_projected_graph(B, user_nodes)

    # Largest connected component for clarity
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

        # Community detection metrics
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))
            modularity  = nx.algorithms.community.quality.modularity(
                G, communities)
            print(f"\nNetwork Metrics:")
            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            print(f"  Communities: {len(communities)}")
            print(f"  Modularity:  {modularity:.4f}")
            print(f"  Avg community size: {np.mean([len(c) for c in communities]):.1f}")
        except Exception as e:
            print(f"  Community detection skipped: {e}")

        # Draw
        plt.figure(figsize=(14, 14))
        pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)
        node_colors = []
        for node in G.nodes():
            if node in user_profiles:
                s = user_profiles[node]['dominant_stance']
                node_colors.append(palette.get(s, '#95a5a6'))
            else:
                node_colors.append('#95a5a6')
        node_sizes = [G.degree(n) * 30 + 50 for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                               node_color=node_colors, alpha=0.85)
        nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.8)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Pro-Palestine',
                   markerfacecolor='#2ecc71', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Pro-Israel',
                   markerfacecolor='#3498db', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Neutral',
                   markerfacecolor='#95a5a6', markersize=12),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        plt.title(f'User Interaction Network (Top {top_n} Users)\n'
                  'Nodes colored by dominant stance; size by degree',
                  fontsize=15, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '24_user_network_graph.png', dpi=300, bbox_inches='tight')
        print("✔ Saved: 04_user_network_graph.png")
        plt.close()
else:
    print("⚠️  Skipping network visualization.")

print("\n" + "=" * 80)
print("✅ NETWORK ANALYSIS COMPLETE")
print("=" * 80)
