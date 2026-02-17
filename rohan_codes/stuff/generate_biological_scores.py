"""
Generate biological scores using refactored extended pipeline
All scores are now properly integrated into the bioscore framework
"""
import sys
import os
import pandas as pd

# Add repo roots
sys.path.append(os.path.abspath("DesignPrinciplesGeneNetworks"))
sys.path.append(os.path.abspath("Optimised-Boolean-Network-Inference/My_code/bioscore_refactored"))

from bioscore.loader import load_meta_analysis_models
from bioscore.pipeline import compute_scores
from bioscore.config import ScoreConfig

# Import the extended registry from bioscore.scores package
from bioscore.scores.extended_registry import build_extended_score_registry


def main():
    print("Loading biological models...")
    os.chdir("DesignPrinciplesGeneNetworks")
    models, meta = load_meta_analysis_models(
        max_degree=12,
        max_N=10000
    )
    print(f"Loaded {len(models)} biological models")
    
    # Build extended registry with all scores
    registry = build_extended_score_registry()
    cfg = ScoreConfig()
    
    print("Computing all scores (standard + extended)...")
    df = compute_scores(models, cfg, registry)
    
    # Save results
    os.chdir("..")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/biological_scores.csv", index=False)
    
    print("\nSaved biological scores with all features:")
    print(f"  - Standard bioscore metrics")
    print(f"  - FFL statistics (total, coherent, incoherent)")
    print(f"  - 3-node feedback loops (total, positive, negative)")
    print(f"  - Distribution histograms (in/out degree, sensitivity, canalizing depth)")
    print(f"\nTotal features: {len(df.columns)}")
    print(df.describe())


if __name__ == "__main__":
    main()