import sys
import os
import numpy as np
import pandas as pd
import json

# ---------------------------------------------------------
# Add repo roots
# ---------------------------------------------------------
sys.path.append(os.path.abspath("DesignPrinciplesGeneNetworks"))
sys.path.append(os.path.abspath("Optimised-Boolean-Network-Inference/My_code/bioscore_refactored"))

from bioscore.loader import load_meta_analysis_models
from bioscore.pipeline import build_score_registry, compute_scores
from bioscore.config import ScoreConfig
import canalizing_function_toolbox_v13 as can

# =========================================================
# Utility functions
# =========================================================
def histogram_vector(values, bins):
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist.tolist()

def compute_node_sensitivity(f):
    n = int(np.log2(len(f)))
    if n == 0:
        return 0.0
    sensitivity = 0.0
    for i in range(n):
        flip_mask = 1 << (n - 1 - i)
        flipped = np.arange(2**n) ^ flip_mask
        sensitivity += np.mean(f != f[flipped])
    return sensitivity

# =========================================================
# FFL computation (robust + correct)
# =========================================================
def compute_ffl_stats(Fs, Is, constantss):
    N = len(Fs)
    total_ffl = np.zeros(N)
    coherent = np.zeros(N)
    incoherent = np.zeros(N)
    
    for i in range(N):
        F = Fs[i]
        I = Is[i]
        
        # Skip networks containing empty functions
        if any(len(f) == 0 for f in F):
            continue
        
        try:
            A = can.adjacency_matrix(I, constantss[i])
            ffls, types = can.get_ffls(A, F, I)
        except:
            continue
        
        for t in types:
            if t == -1 or t == -2:
                continue
            total_ffl[i] += 1
            if can.is_ffl_coherent(t):
                coherent[i] += 1
            else:
                incoherent[i] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        coh_prop = np.where(total_ffl > 0, coherent / total_ffl, 0)
        incoh_prop = np.where(total_ffl > 0, incoherent / total_ffl, 0)
    
    return total_ffl, coh_prop, incoh_prop

# =========================================================
# OPTIMIZED 3-node feedback loops
# =========================================================
def compute_fbl_3node_optimized(Fs, Is):
    """
    Optimized version that finds 3-node cycles directly using adjacency matrix
    instead of NetworkX's simple_cycles
    """
    N = len(Fs)
    total_loops = np.zeros(N)
    pos_loops = np.zeros(N)
    neg_loops = np.zeros(N)
    
    for i in range(N):
        I = Is[i]
        F = Fs[i]
        n_vars = len(I)
        
        # Build adjacency matrix
        adj = np.zeros((n_vars, n_vars), dtype=bool)
        for target, regs in enumerate(I):
            for r in regs:
                if r < n_vars:
                    adj[r, target] = True
        
        # Find all 3-node cycles efficiently
        # A 3-cycle exists from i->j->k->i if adj[i,j] and adj[j,k] and adj[k,i]
        cycles = []
        for i_node in range(n_vars):
            # Find all neighbors of i_node
            j_nodes = np.where(adj[i_node, :])[0]
            for j_node in j_nodes:
                if j_node == i_node:
                    continue
                # Find all neighbors of j_node
                k_nodes = np.where(adj[j_node, :])[0]
                for k_node in k_nodes:
                    if k_node == i_node or k_node == j_node:
                        continue
                    # Check if k_node connects back to i_node
                    if adj[k_node, i_node]:
                        # Found a 3-cycle, but avoid duplicates
                        cycle = tuple(sorted([i_node, j_node, k_node]))
                        if cycle not in cycles:
                            cycles.append(cycle)
        
        # Analyze each cycle
        for cycle in cycles:
            # Convert back to list for the toolbox functions
            loop = list(cycle)
            try:
                t = can.get_type_of_loop(loop, F, I)
                el = can.get_loop_type_number(t)
            except:
                continue
            
            if el < 0:
                continue
            
            total_loops[i] += 1
            if el % 2 == 0:
                pos_loops[i] += 1
            else:
                neg_loops[i] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pos_prop = np.where(total_loops > 0, pos_loops / total_loops, 0)
        neg_prop = np.where(total_loops > 0, neg_loops / total_loops, 0)
    
    return total_loops, pos_prop, neg_prop

# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading biological models...")
    os.chdir("DesignPrinciplesGeneNetworks")
    models, meta = load_meta_analysis_models(
        max_degree=12,
        max_N=10000
    )
    print(f"Loaded {len(models)} biological models")
    
    registry = build_score_registry()
    cfg = ScoreConfig()
    
    print("Computing original bioscore metrics...")
    df = compute_scores(models, cfg, registry)
    
    # -----------------------------------------------------
    # Extract core model components
    # -----------------------------------------------------
    Fs = [m.F for m in models]
    Is = [m.I for m in models]
    constantss = [m.constants for m in models]
    
    # -----------------------------------------------------
    # FFLs
    # -----------------------------------------------------
    print("Computing FFL statistics...")
    total_ffl, coh_prop, incoh_prop = compute_ffl_stats(
        Fs, Is, constantss
    )
    df["total_FFLs"] = total_ffl
    df["coherent_FFL_proportion"] = coh_prop
    df["incoherent_FFL_proportion"] = incoh_prop
    
    # -----------------------------------------------------
    # 3-node Feedback Loops (OPTIMIZED)
    # -----------------------------------------------------
    print("Computing 3-node feedback loops (optimized)...")
    total_fbl, pos_prop, neg_prop = compute_fbl_3node_optimized(
        Fs, Is
    )
    df["total_3node_loops"] = total_fbl
    df["positive_feedback_proportion"] = pos_prop
    df["negative_feedback_proportion"] = neg_prop
    
    # -----------------------------------------------------
    # Distributions
    # -----------------------------------------------------
    print("Computing distributions...")
    in_degree_hists = []
    out_degree_hists = []
    sensitivity_hists = []
    canalizing_depth_hists = []
    
    for m in models:
        F = m.F
        I = m.I
        n_genes = len(F)
        
        # In-degree histogram
        in_degrees = [len(regs) for regs in I]
        max_in = max(in_degrees) if len(in_degrees) > 0 else 1
        in_hist = histogram_vector(in_degrees, bins=range(0, max_in + 2))
        in_degree_hists.append(json.dumps(in_hist))
        
        # Out-degree histogram
        out_degrees = [0] * n_genes
        for target, regs in enumerate(I):
            for r in regs:
                if r < n_genes:
                    out_degrees[r] += 1
        max_out = max(out_degrees) if len(out_degrees) > 0 else 1
        out_hist = histogram_vector(out_degrees, bins=range(0, max_out + 2))
        out_degree_hists.append(json.dumps(out_hist))
        
        # Sensitivity histogram
        node_sens = []
        for f in F:
            if len(f) > 0:
                try:
                    node_sens.append(
                        compute_node_sensitivity(np.array(f))
                    )
                except:
                    continue
        sens_hist = histogram_vector(node_sens, bins=10) if len(node_sens) > 0 else []
        sensitivity_hists.append(json.dumps(sens_hist))
        
        # Canalizing depth histogram
        depths = []
        for f in F:
            if len(f) > 0:
                try:
                    depth, *_ = can.find_layers(np.array(f))
                    depths.append(depth)
                except:
                    continue
        if len(depths) > 0:
            max_depth = max(depths)
            depth_hist = histogram_vector(depths, bins=range(0, max_depth + 2))
        else:
            depth_hist = []
        canalizing_depth_hists.append(json.dumps(depth_hist))
    
    df["in_degree_histogram"] = in_degree_hists
    df["out_degree_histogram"] = out_degree_hists
    df["sensitivity_histogram"] = sensitivity_hists
    df["canalizing_depth_histogram"] = canalizing_depth_hists
    
    os.chdir("..")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/biological_scores.csv", index=False)
    
    print("Saved biological scores with motifs and distributions.")
    print(df.describe())

if __name__ == "__main__":
    main()