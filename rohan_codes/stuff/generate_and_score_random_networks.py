"""
Step 1: Generate Random Networks and Compare Distributions

This script:
1. Loads biological network scores
2. Generates random networks with similar size/degree constraints
3. Computes the same scores for random networks
4. Compares distributions (scalar scores and histogram distances)
5. Outputs visualizations and statistical comparisons
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add repo paths
sys.path.append(os.path.abspath("DesignPrinciplesGeneNetworks"))
sys.path.append(os.path.abspath("Optimised-Boolean-Network-Inference/My_code/bioscore_refactored"))

from bioscore.pipeline import compute_scores
from bioscore.config import ScoreConfig
from bioscore.scores.extended_registry import build_extended_score_registry
import canalizing_function_toolbox_v13 as can

# =========================================================
# Random Network Generation
# =========================================================

# =========================================================
# Simple Model Class (matches bioscore interface)
# =========================================================

class SimpleModel:
    """Minimal model class that matches the bioscore/BooleanModel interface"""
    def __init__(self, F, I, constants, degrees, name, max_degree_used=12):
        # Core attributes
        self.F = F  # Full list including constants
        self.I = I
        self.constants = constants
        self.name = name
        self.n_variables = len(F)
        self.n_constants = sum(constants)
        
        # Degrees
        self.degrees = np.array(degrees)  # Used by dynamics.py
        self.degrees_genes = np.array(degrees)  # Used by structure.py
        
        # For random networks, assume all regulators are essential
        self.degrees_essential_genes = self.degrees_genes.copy()
        
        # F_genes used by canalization functions - same as F for our case
        self.F_genes = F
        
        # Max degree used (needed for some score functions)
        self.max_degree_used = max_degree_used

# =========================================================
# Random Network Generation
# =========================================================

def generate_random_networks(bio_df, n_random=1000):
    """
    Generate random networks matching size/degree of biological networks
    
    Args:
        bio_df: DataFrame of biological network scores
        n_random: Number of random networks to generate
    
    Returns:
        List of SimpleModel objects
    """
    print(f"Generating {n_random} random networks...")
    
    # Get distribution of network sizes and degrees from biological data
    n_genes_dist = bio_df['n_genes'].values
    mean_in_degree_dist = bio_df['mean_in_degree'].values
    
    random_networks = []
    for idx in tqdm(range(n_random), desc="Generating networks", unit="network"):
        # Sample network size and degree from biological distributions
        n_genes = int(np.random.choice(n_genes_dist))
        mean_in_degree = float(np.random.choice(mean_in_degree_dist))
        
        # Generate network structure
        F = []
        I = []
        constants = []
        degrees = []
        
        for i in range(n_genes):
            # Sample in-degree from Poisson distribution around mean
            k = max(0, int(np.random.poisson(mean_in_degree)))
            k = min(k, n_genes)  # Can't exceed network size
            degrees.append(k)
            
            # Randomly select k regulators
            if k > 0:
                regulators = list(np.random.choice(n_genes, size=k, replace=False))
            else:
                regulators = []
            
            I.append(regulators)
            
            # Generate random Boolean function
            if k > 0:
                truth_table = list(np.random.randint(0, 2, size=2**k))
            else:
                # No inputs -> constant function
                truth_table = [np.random.randint(0, 2)]
            
            F.append(truth_table)
            
            # Mark as constant if no inputs or constant output
            is_constant = (len(regulators) == 0) or (len(set(truth_table)) == 1)
            constants.append(is_constant)
        
        model = SimpleModel(
            F=F, 
            I=I, 
            constants=constants,
            degrees=degrees,
            name=f"random_{idx}",
            max_degree_used=12
        )
        random_networks.append(model)
    
    print(f"Generated {n_random} random networks.")
    return random_networks

# =========================================================
# Score Computation for Random Networks
# =========================================================

def compute_scores_for_random_networks(networks):
    """
    Compute all scores for random networks using refactored extended pipeline
    
    Args:
        networks: List of SimpleModel objects
    
    Returns:
        DataFrame with all scores (standard + extended)
    """
    print("Computing scores for random networks...")
    
    # Use the extended registry that includes all scores
    registry = build_extended_score_registry()
    cfg = ScoreConfig()
    
    # Compute scores with progress bar
    results = []
    for model in tqdm(networks, desc="Scoring networks", unit="network"):
        row = {"model_name": model.name}
        for spec in registry:
            try:
                res = spec.fn(model, cfg)
                row[res.name] = res.value
            except Exception as e:
                row[spec.name] = np.nan
        results.append(row)
    
    df = pd.DataFrame(results)
    
    print(f"  Computed {len(df.columns)} features for {len(df)} networks")
    
    return df

# =========================================================
# Distribution Comparison
# =========================================================

def compare_scalar_distributions(bio_df, random_df, output_dir):
    """
    Compare distributions of network-wide scalar scores
    
    Creates violin plots and computes statistical tests
    """
    print("\nComparing scalar score distributions...")
    
    # Scalar columns (exclude model_name and histograms)
    scalar_cols = [col for col in bio_df.columns 
                   if col not in ['model_name', 'in_degree_histogram', 
                                  'out_degree_histogram', 'sensitivity_histogram',
                                  'canalizing_depth_histogram']]
    
    results = []
    
    # Create output directory for plots
    scalar_dir = os.path.join(output_dir, 'scalar_comparisons')
    os.makedirs(scalar_dir, exist_ok=True)
    
    for col in tqdm(scalar_cols, desc="Comparing scalar features", unit="feature"):
        bio_vals = bio_df[col].dropna().values
        random_vals = random_df[col].dropna().values
        
        if len(bio_vals) == 0 or len(random_vals) == 0:
            continue
        
        # Statistical tests
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(bio_vals, random_vals, alternative='two-sided')
        
        # Effect size: Cohen's d
        cohens_d = (np.mean(bio_vals) - np.mean(random_vals)) / np.sqrt(
            (np.std(bio_vals)**2 + np.std(random_vals)**2) / 2
        )
        
        # KS test
        ks_stat, ks_p = stats.ks_2samp(bio_vals, random_vals)
        
        results.append({
            'feature': col,
            'bio_mean': np.mean(bio_vals),
            'bio_std': np.std(bio_vals),
            'random_mean': np.mean(random_vals),
            'random_std': np.std(random_vals),
            'mann_whitney_p': p_value,
            'cohens_d': cohens_d,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'discriminative': p_value < 0.01  # Significant difference
        })
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_to_plot = pd.DataFrame({
            'Value': np.concatenate([bio_vals, random_vals]),
            'Type': ['Biological']*len(bio_vals) + ['Random']*len(random_vals)
        })
        
        sns.violinplot(data=data_to_plot, x='Type', y='Value', ax=ax)
        ax.set_title(f'{col}\np={p_value:.2e}, Cohen\'s d={cohens_d:.3f}')
        ax.set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(scalar_dir, f'{col}.png'), dpi=150)
        plt.close()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mann_whitney_p')
    results_df.to_csv(os.path.join(output_dir, 'scalar_comparison_results.csv'), index=False)
    
    print(f"  Analyzed {len(results)} scalar features")
    print(f"  {sum(results_df['discriminative'])} features significantly discriminate (p<0.01)")
    
    return results_df

def compute_distribution_distances(hists1, hists2, metric='jensenshannon'):
    """
    Compute pairwise distances between distributions
    
    Args:
        hists1, hists2: Lists of histograms (as JSON strings)
        metric: 'jensenshannon' or 'wasserstein'
    
    Returns:
        Array of distances
    """
    distances = []
    
    for h1_str, h2_str in zip(hists1, hists2):
        try:
            h1 = np.array(json.loads(h1_str))
            h2 = np.array(json.loads(h2_str))
            
            if len(h1) == 0 or len(h2) == 0:
                continue
            
            # Pad to same length
            max_len = max(len(h1), len(h2))
            h1_padded = np.pad(h1, (0, max_len - len(h1)), 'constant')
            h2_padded = np.pad(h2, (0, max_len - len(h2)), 'constant')
            
            # Normalize to probabilities
            h1_norm = h1_padded / (h1_padded.sum() + 1e-10)
            h2_norm = h2_padded / (h2_padded.sum() + 1e-10)
            
            if metric == 'jensenshannon':
                dist = jensenshannon(h1_norm, h2_norm)
            elif metric == 'wasserstein':
                dist = wasserstein_distance(h1_norm, h2_norm)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if not np.isnan(dist):
                distances.append(dist)
        except:
            continue
    
    return np.array(distances)

def compare_histogram_distributions(bio_df, random_df, output_dir, n_samples=500):
    """
    Compare histogram features using distribution-to-distribution distances
    
    For each histogram feature, we compute:
    - Within-biological distances (nC2 pairs)
    - Within-random distances (nC2 pairs)
    - Between-group distances (bio vs random)
    """
    print("\nComparing histogram distributions...")
    
    hist_cols = ['in_degree_histogram', 'out_degree_histogram', 
                 'sensitivity_histogram', 'canalizing_depth_histogram']
    
    results = []
    hist_dir = os.path.join(output_dir, 'histogram_comparisons')
    os.makedirs(hist_dir, exist_ok=True)
    
    for col in tqdm(hist_cols, desc="Comparing histogram features", unit="feature"):
        bio_hists = bio_df[col].values
        random_hists = random_df[col].values
        
        # Sample pairs to avoid combinatorial explosion
        n_bio = len(bio_hists)
        n_random = len(random_hists)
        
        # Within-biological distances
        bio_pairs = list(combinations(range(n_bio), 2))
        if len(bio_pairs) > n_samples:
            bio_pairs = [bio_pairs[i] for i in np.random.choice(len(bio_pairs), n_samples, replace=False)]
        
        within_bio_dists = []
        for i, j in bio_pairs:
            dists = compute_distribution_distances([bio_hists[i]], [bio_hists[j]])
            if len(dists) > 0:
                within_bio_dists.append(dists[0])
        
        # Within-random distances
        random_pairs = list(combinations(range(n_random), 2))
        if len(random_pairs) > n_samples:
            random_pairs = [random_pairs[i] for i in np.random.choice(len(random_pairs), n_samples, replace=False)]
        
        within_random_dists = []
        for i, j in random_pairs:
            dists = compute_distribution_distances([random_hists[i]], [random_hists[j]])
            if len(dists) > 0:
                within_random_dists.append(dists[0])
        
        # Between-group distances (bio vs random)
        between_indices = [(np.random.randint(n_bio), np.random.randint(n_random)) 
                          for _ in range(n_samples)]
        
        between_dists = []
        for i, j in between_indices:
            dists = compute_distribution_distances([bio_hists[i]], [random_hists[j]])
            if len(dists) > 0:
                between_dists.append(dists[0])
        
        # Statistical comparison
        # Are between-group distances larger than within-group?
        if len(between_dists) > 0 and len(within_bio_dists) > 0:
            _, p_bio_vs_between = stats.mannwhitneyu(within_bio_dists, between_dists, alternative='two-sided')
        else:
            p_bio_vs_between = np.nan
        
        if len(between_dists) > 0 and len(within_random_dists) > 0:
            _, p_random_vs_between = stats.mannwhitneyu(within_random_dists, between_dists, alternative='two-sided')
        else:
            p_random_vs_between = np.nan
        
        results.append({
            'feature': col,
            'within_bio_mean': np.mean(within_bio_dists) if len(within_bio_dists) > 0 else np.nan,
            'within_bio_std': np.std(within_bio_dists) if len(within_bio_dists) > 0 else np.nan,
            'within_random_mean': np.mean(within_random_dists) if len(within_random_dists) > 0 else np.nan,
            'within_random_std': np.std(within_random_dists) if len(within_random_dists) > 0 else np.nan,
            'between_mean': np.mean(between_dists) if len(between_dists) > 0 else np.nan,
            'between_std': np.std(between_dists) if len(between_dists) > 0 else np.nan,
            'p_bio_vs_between': p_bio_vs_between,
            'p_random_vs_between': p_random_vs_between,
            'discriminative': (p_bio_vs_between < 0.01 or p_random_vs_between < 0.01)
        })
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_dists = []
        labels = []
        
        if len(within_bio_dists) > 0:
            all_dists.extend(within_bio_dists)
            labels.extend(['Within Bio']*len(within_bio_dists))
        
        if len(within_random_dists) > 0:
            all_dists.extend(within_random_dists)
            labels.extend(['Within Random']*len(within_random_dists))
        
        if len(between_dists) > 0:
            all_dists.extend(between_dists)
            labels.extend(['Between Groups']*len(between_dists))
        
        data_to_plot = pd.DataFrame({
            'Distance': all_dists,
            'Comparison': labels
        })
        
        sns.violinplot(data=data_to_plot, x='Comparison', y='Distance', ax=ax)
        ax.set_title(f'{col}\nDistance Comparisons')
        ax.set_ylabel('Jensen-Shannon Distance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, f'{col}_distances.png'), dpi=150)
        plt.close()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'histogram_comparison_results.csv'), index=False)
    
    print(f"  Analyzed {len(results)} histogram features")
    print(f"  {sum(results_df['discriminative'])} features significantly discriminate")
    
    return results_df

# =========================================================
# Summary Visualization
# =========================================================

def create_summary_report(scalar_results, hist_results, output_dir):
    """Create summary visualization of discriminative power"""
    
    print("\nCreating summary report...")
    
    # Combine results
    all_results = []
    
    for _, row in scalar_results.iterrows():
        all_results.append({
            'Feature': row['feature'],
            'Type': 'Scalar',
            'p_value': row['mann_whitney_p'],
            'effect_size': abs(row['cohens_d'])
        })
    
    for _, row in hist_results.iterrows():
        all_results.append({
            'Feature': row['feature'],
            'Type': 'Histogram',
            'p_value': min(row['p_bio_vs_between'], row['p_random_vs_between']),
            'effect_size': row['between_mean'] - min(row['within_bio_mean'], row['within_random_mean'])
        })
    
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('p_value')
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # P-value comparison
    colors = ['blue' if t == 'Scalar' else 'orange' for t in summary_df['Type']]
    ax1.barh(range(len(summary_df)), -np.log10(summary_df['p_value']), color=colors)
    ax1.set_yticks(range(len(summary_df)))
    ax1.set_yticklabels(summary_df['Feature'], fontsize=8)
    ax1.axvline(-np.log10(0.01), color='red', linestyle='--', label='p=0.01')
    ax1.set_xlabel('-log10(p-value)')
    ax1.set_title('Statistical Significance')
    ax1.legend()
    ax1.invert_yaxis()
    
    # Effect size
    ax2.barh(range(len(summary_df)), summary_df['effect_size'], color=colors)
    ax2.set_yticks(range(len(summary_df)))
    ax2.set_yticklabels(summary_df['Feature'], fontsize=8)
    ax2.set_xlabel('Effect Size')
    ax2.set_title('Effect Size (|Cohen\'s d| or Distance Diff)')
    ax2.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Scalar'),
                      Patch(facecolor='orange', label='Histogram')]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_discriminative_power.png'), dpi=200)
    plt.close()
    
    # Save summary
    summary_df.to_csv(os.path.join(output_dir, 'summary_all_features.csv'), index=False)
    
    print(f"  Total features: {len(summary_df)}")
    print(f"  Significant (p<0.01): {sum(summary_df['p_value'] < 0.01)}")
    print(f"  Top 5 discriminative features:")
    print(summary_df.head(5)[['Feature', 'Type', 'p_value', 'effect_size']])

# =========================================================
# MAIN
# =========================================================

def main():
    print("="*60)
    print("Step 1: Generate Random Networks & Compare Distributions")
    print("="*60)
    
    # Load biological scores
    print("\nLoading biological network scores...")
    bio_df = pd.read_csv("data/biological_scores.csv")
    print(f"Loaded {len(bio_df)} biological networks")
    
    # Generate random networks
    n_random = 1000
    random_networks = generate_random_networks(bio_df, n_random=n_random)
    
    # Compute scores for random networks
    random_df = compute_scores_for_random_networks(random_networks)
    
    # Save random network scores
    os.makedirs("data", exist_ok=True)
    random_df.to_csv("data/random_scores.csv", index=False)
    print(f"\nSaved random network scores to data/random_scores.csv")
    
    # Create output directory
    output_dir = "results/step1_distribution_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare scalar distributions
    scalar_results = compare_scalar_distributions(bio_df, random_df, output_dir)
    
    # Compare histogram distributions
    hist_results = compare_histogram_distributions(bio_df, random_df, output_dir, n_samples=500)
    
    # Create summary report
    create_summary_report(scalar_results, hist_results, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()