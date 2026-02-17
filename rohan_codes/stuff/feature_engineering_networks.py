"""
Simplified Feature Engineering
- Excludes cryptic derived metrics
- Minimal, clean plots
- Focus on interpretable features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import json
import warnings
warnings.filterwarnings('ignore')

# Clean plotting
sns.set_style("ticks")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

def save_fig(name):
    plt.savefig(f"results/{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# =========================================================
# LOAD & CURATE
# =========================================================

print("\n" + "="*70)
print("FEATURE ENGINEERING - SIMPLIFIED")
print("="*70)

bio = pd.read_csv("data/biological_scores.csv")
pool = pd.read_csv("data/random_pool_10k.csv")

# Define what to exclude - only obvious non-features
EXCLUDE = [
    'model_name',  # Not a feature
    'n_genes', 'n_constants',  # Size - we'll constrain nulls to match this
    'total_FFLs', 'total_3node_loops',  # Raw counts - size dependent
]

HISTOGRAMS = ['in_degree_histogram', 'out_degree_histogram', 
              'sensitivity_histogram', 'canalizing_depth_histogram']

# Keep ALL meaningful features (including those Ke/bias metrics)
network_features = [c for c in bio.columns if c not in EXCLUDE + HISTOGRAMS]

print(f"\nAll features to test: {len(network_features)}")
for f in network_features:
    print(f"  {f}")

import os
os.makedirs("results", exist_ok=True)

# =========================================================
# 1. BASELINE DISCRIMINATION
# =========================================================

print("\n[1/3] Testing discrimination...")

baseline = []
for feat in network_features:
    bio_vals = bio[feat].dropna().values
    pool_vals = pool[feat].dropna().values
    
    if len(bio_vals) < 5 or len(pool_vals) < 5:
        continue
    
    _, p = stats.mannwhitneyu(bio_vals, pool_vals, alternative='two-sided')
    d = (bio_vals.mean() - pool_vals.mean()) / np.sqrt((bio_vals.std()**2 + pool_vals.std()**2)/2)
    
    baseline.append({
        'feature': feat,
        'p_value': p,
        'cohens_d': d,
        'discriminative': p < 0.01
    })

baseline_df = pd.DataFrame(baseline).sort_values('cohens_d', key=abs, ascending=False)
discrim_features = baseline_df[baseline_df['discriminative']]['feature'].tolist()

print(f"\nDiscriminative: {len(discrim_features)}/{len(baseline_df)}")

# =========================================================
# 2. CORRELATION & CONSTRAINT TESTING
# =========================================================

print("\n[2/3] Testing robustness...")

# Find high correlations
corr = bio[discrim_features].corr(method='pearson').fillna(0)
high_corr = []
for i in range(len(discrim_features)):
    for j in range(i+1, len(discrim_features)):
        r = corr.iloc[i, j]
        if abs(r) > 0.7:
            high_corr.append((discrim_features[i], discrim_features[j], r))

print(f"\nHigh correlations (|r|>0.7): {len(high_corr)}")
for f1, f2, r in high_corr:
    print(f"  {f1:35s} ↔ {f2:35s}  r={r:+.2f}")

# Constraint testing
retention = []
constraint_details = []  # Track which constraints hurt which features

for test_feat in discrim_features:
    bio_test = bio[test_feat].dropna().values
    retains = []
    
    for const_feat in discrim_features:
        if test_feat == const_feat:
            continue
        
        bio_const = bio[const_feat].dropna().values
        if len(bio_const) < 5:
            continue
        
        p10, p90 = np.percentile(bio_const, [10, 90])
        constrained = pool[(pool[const_feat] >= p10) & (pool[const_feat] <= p90)]
        
        if len(constrained) < 50:
            continue
        
        const_test = constrained[test_feat].dropna().values
        try:
            _, p = stats.mannwhitneyu(bio_test, const_test, alternative='two-sided')
            still_discrim = p < 0.01
            retains.append(still_discrim)
            
            # Track if this constraint hurt discrimination
            constraint_details.append({
                'tested': test_feat,
                'constrained': const_feat,
                'retains': still_discrim
            })
        except:
            pass
    
    if len(retains) > 0:
        retention.append({
            'feature': test_feat,
            'retention_rate': np.mean(retains)
        })

retention_df = pd.DataFrame(retention).sort_values('retention_rate', ascending=False)
constraint_df = pd.DataFrame(constraint_details)

# =========================================================
# 3. SELECT FINAL FEATURES
# =========================================================

print("\n[3/3] Selecting final features...")

# Remove complementary pairs
to_remove = set()
complementary = [
    ('coherent_FFL_proportion', 'incoherent_FFL_proportion'),
    ('positive_feedback_proportion', 'negative_feedback_proportion')
]

for f1, f2 in complementary:
    if f1 in discrim_features and f2 in discrim_features:
        d1 = baseline_df[baseline_df['feature']==f1]['cohens_d'].abs().iloc[0]
        d2 = baseline_df[baseline_df['feature']==f2]['cohens_d'].abs().iloc[0]
        to_remove.add(f2 if d1 > d2 else f1)

# Robust features
robust = retention_df[retention_df['retention_rate'] > 0.5]['feature'].tolist()
robust = [f for f in robust if f not in to_remove]

# Rank by combined score
final_scores = []
for feat in robust:
    d = baseline_df[baseline_df['feature']==feat]['cohens_d'].abs().iloc[0]
    ret = retention_df[retention_df['feature']==feat]['retention_rate'].iloc[0]
    final_scores.append({'feature': feat, 'score': d * ret, 'd': d, 'ret': ret})

final_df = pd.DataFrame(final_scores).sort_values('score', ascending=False)
final_network = final_df.head(6)['feature'].tolist()

# Histograms
hist_results = []
for h in HISTOGRAMS:
    bio_h = bio[h].values
    pool_h = pool[h].values
    
    def js(h1, h2):
        try:
            h1 = np.array(json.loads(h1))
            h2 = np.array(json.loads(h2))
            if len(h1)==0 or len(h2)==0: return np.nan
            m = max(len(h1), len(h2))
            h1 = np.pad(h1, (0, m-len(h1)))
            h2 = np.pad(h2, (0, m-len(h2)))
            h1 = h1/(h1.sum()+1e-10)
            h2 = h2/(h2.sum()+1e-10)
            return jensenshannon(h1, h2)
        except:
            return np.nan
    
    within = [js(bio_h[i], bio_h[j]) for i, j in 
              [(np.random.randint(len(bio_h)), np.random.randint(len(bio_h))) for _ in range(100)]]
    between = [js(bio_h[i], pool_h[j]) for i, j in 
               [(np.random.randint(len(bio_h)), np.random.randint(len(pool_h))) for _ in range(100)]]
    
    within = [x for x in within if not np.isnan(x)]
    between = [x for x in between if not np.isnan(x)]
    
    if len(within) > 0 and len(between) > 0:
        _, p = stats.mannwhitneyu(within, between, alternative='less')
        hist_results.append({'feature': h, 'discriminative': p < 0.01})

final_histograms = [r['feature'] for r in hist_results if r['discriminative']]

# =========================================================
# OUTPUT
# =========================================================

print("\n" + "="*70)
print("FINAL FEATURES")
print("="*70)

print(f"\nNetwork ({len(final_network)}):")
for _, row in final_df.head(6).iterrows():
    print(f"  {row['feature']:40s}  d={row['d']:+.2f}  ret={row['ret']:.0%}")

print(f"\nHistograms ({len(final_histograms)}):")
for h in final_histograms:
    print(f"  {h}")

# Save
result = {'network_wide': final_network, 'histograms': final_histograms}
with open("results/final_features.json", 'w') as f:
    json.dump(result, f, indent=2)
with open("results/final_features.txt", 'w') as f:
    for feat in final_network + final_histograms:
        f.write(f"{feat}\n")

# =========================================================
# MINIMAL PLOTS
# =========================================================

# Plot 1: Violin plots showing constraint effects
print("\nCreating comparison plots...")

# Get features with lowest retention rates (even if they passed >50%)
# These show the MOST impact from constraints
retention_sorted = retention_df.sort_values('retention_rate')
features_to_plot = retention_sorted.head(3)['feature'].tolist()

print(f"  Plotting features most affected by constraints:")
for feat in features_to_plot:
    ret = retention_df[retention_df['feature']==feat]['retention_rate'].iloc[0]
    print(f"    {feat}: {ret:.0%} retention")

if len(features_to_plot) > 0:
    fig, axes = plt.subplots(len(features_to_plot), 2, figsize=(10, 3*len(features_to_plot)))
    if len(features_to_plot) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, feat in enumerate(features_to_plot):
        # Find which constraint caused the biggest drop in discrimination
        best_constraint = None
        best_p_diff = 0
        
        # Get baseline p-value
        p_baseline = baseline_df[baseline_df['feature']==feat]['p_value'].iloc[0]
        
        for const_feat in discrim_features:
            if const_feat == feat:
                continue
            
            # Test this constraint
            bio_const = bio[const_feat].dropna().values
            if len(bio_const) < 5:
                continue
            
            p10, p90 = np.percentile(bio_const, [10, 90])
            constrained = pool[(pool[const_feat] >= p10) & (pool[const_feat] <= p90)]
            
            if len(constrained) < 50:
                continue
            
            bio_test = bio[feat].dropna().values
            const_test = constrained[feat].dropna().values
            
            try:
                _, p_const = stats.mannwhitneyu(bio_test, const_test, alternative='two-sided')
                p_diff = p_const - p_baseline  # How much p-value increased
                
                if p_diff > best_p_diff:
                    best_p_diff = p_diff
                    best_constraint = const_feat
            except:
                pass
        
        if best_constraint:
            ax1, ax2 = axes[idx, 0], axes[idx, 1]
            
            # Left: Bio vs Null (no constraint) - SHOULD discriminate
            bio_vals = bio[feat].dropna().values
            null_vals = pool[feat].dropna().values[:len(bio_vals)*2]
            
            data1 = pd.DataFrame({
                'Value': np.concatenate([bio_vals, null_vals]),
                'Type': ['Bio']*len(bio_vals) + ['Null']*len(null_vals)
            })
            
            sns.violinplot(data=data1, x='Type', y='Value', ax=ax1, palette=['#2E7D32', '#757575'])
            _, p1 = stats.mannwhitneyu(bio_vals, null_vals, alternative='two-sided')
            ax1.set_title(f'{feat}\n(Unconstrained, p={p1:.1e})', fontsize=9, fontweight='bold')
            ax1.set_xlabel('')
            ax1.set_ylabel('Value' if idx == len(features_to_plot)//2 else '', fontweight='bold', fontsize=9)
            
            # Right: Bio vs Constrained Null - LOSES discrimination
            bio_const = bio[best_constraint].dropna().values
            p10, p90 = np.percentile(bio_const, [10, 90])
            constrained_pool = pool[(pool[best_constraint] >= p10) & (pool[best_constraint] <= p90)]
            const_vals = constrained_pool[feat].dropna().values[:len(bio_vals)*2]
            
            if len(const_vals) > 10:
                data2 = pd.DataFrame({
                    'Value': np.concatenate([bio_vals, const_vals]),
                    'Type': ['Bio']*len(bio_vals) + ['Constrained']*len(const_vals)
                })
                
                sns.violinplot(data=data2, x='Type', y='Value', ax=ax2, palette=['#2E7D32', '#F57C00'])
                _, p2 = stats.mannwhitneyu(bio_vals, const_vals, alternative='two-sided')
                const_name = best_constraint[:25] + '...' if len(best_constraint) > 25 else best_constraint
                ax2.set_title(f'{feat}\n(Constrained: {const_name}, p={p2:.1e})', fontsize=9, fontweight='bold')
                ax2.set_xlabel('')
                ax2.set_ylabel('')
    
    sns.despine()
    plt.tight_layout()
    save_fig('comparisons')
    
    print(f"  ✓ Created comparison plots for {len(features_to_plot)} eliminated features")
else:
    print("  (No low-retention features to plot)")

# Plot 2: Simple pipeline
fig, ax = plt.subplots(figsize=(8, 4))
for _, row in final_df.iterrows():
    color = '#2E7D32' if row['feature'] in final_network else '#BDBDBD'
    size = 150 if row['feature'] in final_network else 60
    ax.scatter(row['d'], row['ret'], s=size, c=color, alpha=0.8, edgecolors='black', linewidth=1)

ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
ax.set_xlabel('Effect Size |Cohen\'s d|', fontweight='bold')
ax.set_ylabel('Retention Rate', fontweight='bold')
ax.set_title('Final Feature Selection', fontweight='bold', pad=10)
sns.despine()
ax.grid(alpha=0.2)
save_fig('feature_selection')

# Plot 2: Simple pipeline flow
stages = ['All', 'Discriminative', 'Robust', 'Final']
counts = [len(network_features), len(discrim_features), len(robust), len(final_network)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(stages, counts, color=['#757575', '#1976D2', '#F57C00', '#2E7D32'], 
       edgecolor='black', linewidth=1.5, alpha=0.85)
for i, c in enumerate(counts):
    ax.text(i, c+0.5, str(c), ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Features', fontweight='bold')
ax.set_title('Feature Selection Pipeline', fontweight='bold', pad=10)
sns.despine()
save_fig('pipeline')

print(f"\n✓ Saved to results/")
print("="*70)