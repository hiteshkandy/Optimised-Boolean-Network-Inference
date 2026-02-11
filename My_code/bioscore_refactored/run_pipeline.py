from __future__ import annotations
from bioscore.config import ScoreConfig
from bioscore.loader import load_meta_analysis_models
from bioscore.pipeline import build_score_registry, compute_scores
from bioscore.scores.motifs import compute_motif_cache

def main():
    cfg = ScoreConfig(max_degree=12, derrida_nsim=2000, seed=0)

    models, meta = load_meta_analysis_models(max_degree=cfg.max_degree, max_N=10_000)
    df = compute_scores(models, cfg, build_score_registry())

    # Append motif scores computed once across the dataset
    raw_out = meta["raw_out"]
    Fs = raw_out[0]; Is = raw_out[1]; degrees = raw_out[2]
    variabless = raw_out[4]; constantss = raw_out[5]
    n_variables = raw_out[10]
    N = len(Fs)
    max_degree_used = meta["max_degree_used"]

    motif_cache = compute_motif_cache(
        Fs=Fs, Is=Is, degrees=degrees,
        constantss=constantss, variabless=variabless,
        n_variables=n_variables, N=N, max_degree_used=max_degree_used
    )
    for k, arr in motif_cache.items():
        df[k] = arr

    functional_cols = [
        "mean_canalizing_depth",
        "fraction_nested_canalization",
        "mean_canalizing_strength",
        "mean_input_redundancy",
        "mean_effective_connectivity",
        "mean_bias_p1_minus_p",
        "derrida_value",
    ]
    network_cols = [
        "n_genes", "n_constants",
        "mean_in_degree", "mean_essential_in_degree",
        "cov_Ke_bias", "Ke_times_p1_minus_p", "Ke_times_p1_minus_p_plus_cov", "K_times_p1_minus_p",
        "FBL_pos_count", "FBL_neg_count", "FBL_pos_over_exp_null2", "FBL_neg_over_exp_null2",
        "FFL_coherent_count", "FFL_incoherent_count", "FFL_coherent_over_exp", "FFL_incoherent_over_exp",
    ]

    corr_functional = df[functional_cols].corr(method="pearson")
    corr_network = df[network_cols].corr(method="pearson")

    df.to_csv("all_scores_per_network.csv", index=False)
    corr_functional.to_csv("corr_functional_scores.csv")
    corr_network.to_csv("corr_network_scores_with_motifs.csv")

    print("Wrote: all_scores_per_network.csv, corr_functional_scores.csv, corr_network_scores_with_motifs.csv")

if __name__ == "__main__":
    main()
