"""
Phase 5: Network Analysis
1. User Interaction Networks (Bipartite User-Post Graph)
2. Echo Chamber Detection (Homophily Index)
3. Information Flow Patterns (Temporal Stance Shifts)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('network_output', exist_ok=True)

print("="*80)
print("PHASE 5: NETWORK ANALYSIS & ECHO CHAMBERS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df = pd.read_csv('sentiment_output/reddit_with_sentiment.csv')

# Filter for valid labels
valid_labels = ['P', 'I', 'N']
reddit_df = reddit_df[reddit_df['Label'].isin(valid_labels)]

print(f"✓ Reddit Data: {len(reddit_df)} rows")

# ============================================================================
# 2. USER STANCE PROFILING
# ============================================================================
print("\n" + "-"*60)
print("USER STANCE PROFILING")
print("-"*(60))

if 'author_name' in reddit_df.columns:
    # Count comments per user
    user_counts = reddit_df['author_name'].value_counts()
    active_users = user_counts[user_counts >= 3].index # Users with at least 3 comments
    
    print(f"Total Users: {len(user_counts)}")
    print(f"Active Users (>=3 comments): {len(active_users)}")
    
    # Determine "Dominant Stance" for each active user
    user_profiles = {}
    
    for user in active_users:
        user_comments = reddit_df[reddit_df['author_name'] == user]
        stance_counts = user_comments['Label'].value_counts()
        dominant_stance = stance_counts.idxmax()
        consistency = stance_counts.max() / len(user_comments)
        
        user_profiles[user] = {
            'dominant_stance': dominant_stance,
            'consistency': consistency,
            'total_comments': len(user_comments)
        }
    
    profiles_df = pd.DataFrame.from_dict(user_profiles, orient='index')
    
    # Visualize User Stance Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='dominant_stance', data=profiles_df, palette={'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'})
    plt.title('Dominant Stance of Active Users (Reddit)', fontsize=14, fontweight='bold')
    plt.xlabel('Stance')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig('network_output/01_user_stance_distribution.png', dpi=300)
    print("✓ Saved: 01_user_stance_distribution.png")
    
    # Visualize Consistency
    plt.figure(figsize=(10, 6))
    sns.histplot(data=profiles_df, x='consistency', hue='dominant_stance', element='step', palette={'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'})
    plt.title('User Stance Consistency (1.0 = Always same stance)', fontsize=14, fontweight='bold')
    plt.xlabel('Consistency Score')
    plt.tight_layout()
    plt.savefig('network_output/02_user_consistency.png', dpi=300)
    print("✓ Saved: 02_user_consistency.png")

else:
    print("⚠️ Author name missing. Skipping user profiling.")
    active_users = []

# ============================================================================
# 3. ECHO CHAMBER DETECTION (HOMOPHILY)
# ============================================================================
print("\n" + "-"*60)
print("ECHO CHAMBER DETECTION (HOMOPHILY)")
print("-"*(60))

if len(active_users) > 0 and 'post_id' in reddit_df.columns:
    # 1. Determine "Thread Stance" (Majority stance of comments in a thread)
    thread_stance = reddit_df.groupby('post_id')['Label'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'N')
    
    # 2. Calculate Homophily for each user
    # Homophily = % of comments on threads that match the user's dominant stance
    
    homophily_scores = []
    
    for user in active_users:
        user_dom_stance = user_profiles[user]['dominant_stance']
        user_comments = reddit_df[reddit_df['author_name'] == user]
        
        # Get the stance of the threads they commented on
        thread_ids = user_comments['post_id'].unique()
        matching_threads = 0
        total_threads = 0
        
        for tid in thread_ids:
            if tid in thread_stance:
                t_stance = thread_stance[tid]
                total_threads += 1
                if t_stance == user_dom_stance:
                    matching_threads += 1
        
        if total_threads > 0:
            score = matching_threads / total_threads
            homophily_scores.append({
                'user': user,
                'dominant_stance': user_dom_stance,
                'homophily_index': score
            })
            
    homophily_df = pd.DataFrame(homophily_scores)
    
    print(f"Average Homophily Index: {homophily_df['homophily_index'].mean():.4f}")
    
    # Visualize Homophily by Stance
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dominant_stance', y='homophily_index', data=homophily_df, palette={'P': '#2ecc71', 'I': '#3498db', 'N': '#95a5a6'})
    plt.title('Echo Chamber Effect: Homophily Index by Stance', fontsize=14, fontweight='bold')
    plt.ylabel('Homophily Index (% interaction with same-stance threads)')
    plt.xlabel('User Stance')
    plt.tight_layout()
    plt.savefig('network_output/03_echo_chamber_index.png', dpi=300)
    print("✓ Saved: 03_echo_chamber_index.png")
    
    # Save high homophily users
    echo_chamber_users = homophily_df[homophily_df['homophily_index'] > 0.8]
    print(f"Users in 'Echo Chambers' (>80% same-stance interaction): {len(echo_chamber_users)} ({len(echo_chamber_users)/len(homophily_df):.1%})")

else:
    print("⚠️ Cannot calculate homophily (missing post_id or active users).")

# ============================================================================
# 4. NETWORK VISUALIZATION (BIPARTITE PROJECTION)
# ============================================================================
print("\n" + "-"*60)
print("NETWORK VISUALIZATION")
print("-"*(60))

if len(active_users) > 0 and 'post_id' in reddit_df.columns:
    # Create a graph of Users connected to Threads
    # We'll filter for top 50 most active users to keep the graph readable
    top_50_users = user_counts.head(50).index
    subset_df = reddit_df[reddit_df['author_name'].isin(top_50_users)]
    
    B = nx.Graph()
    B.add_nodes_from(subset_df['author_name'].unique(), bipartite=0, type='user')
    B.add_nodes_from(subset_df['post_id'].unique(), bipartite=1, type='post')
    
    # Add edges
    edges = list(zip(subset_df['author_name'], subset_df['post_id']))
    B.add_edges_from(edges)
    
    # Project to User-User graph (Users connected if they commented on same post)
    user_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'user'}
    G = nx.bipartite.weighted_projected_graph(B, user_nodes)
    
    # Draw Graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Color nodes by stance
    node_colors = []
    for node in G.nodes():
        stance = user_profiles[node]['dominant_stance']
        if stance == 'P': node_colors.append('#2ecc71')
        elif stance == 'I': node_colors.append('#3498db')
        else: node_colors.append('#95a5a6')
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Pro-Palestine', markerfacecolor='#2ecc71', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Pro-Israel', markerfacecolor='#3498db', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutral', markerfacecolor='#95a5a6', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('User Interaction Network (Top 50 Users)\nConnected if commented on same post', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('network_output/04_user_network_graph.png', dpi=300)
    print("✓ Saved: 04_user_network_graph.png")

print("\n" + "="*80)
print("✅ NETWORK ANALYSIS COMPLETE")
print("="*80)
