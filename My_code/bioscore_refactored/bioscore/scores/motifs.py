from __future__ import annotations
import numpy as np
import itertools

import analyse_database13 as ad
import canalizing_function_toolbox_v16 as can

from bioscore.validation import safe_div

def compute_motif_cache(Fs, Is, degrees, constantss, variabless, n_variables, N, max_degree_used):
    """Compute all motif-related arrays ONCE across dataset; return dict of per-model arrays."""
    # Some analyse_database13 motif functions depend on globals
    ad.n_variables = n_variables
    ad.N = N

    # ---- FBLs ----
    (
        nr_loops,
        nr_pos_loops,          # (6, N)
        nr_neg_loops,          # (6, N)
        nr_unknown_loops,
        nr_notreal_loops,
        nr_specific_k_loops,   # (6, 7, N)
        nr_real_loops,
        all_types,
        all_loops
    ) = ad.compute_all_FBLs(Fs, Is, degrees, constantss, max_degree=max_degree_used)

    nr_pos_loops = np.asarray(nr_pos_loops, dtype=float)
    nr_neg_loops = np.asarray(nr_neg_loops, dtype=float)
    nr_specific_k_loops = np.asarray(nr_specific_k_loops, dtype=float)

    obs_pos_total = np.nansum(nr_pos_loops, axis=0)
    obs_neg_total = np.nansum(nr_neg_loops, axis=0)

    (
        prop_pos_separate2,               # (N,)
        _prop_pos_separate_by_scc,
        prop_pos_specific_scc_of_loop,    # (6,7,N)
        _prop_pos_specific_scc_of_loop2
    ) = ad.get_null_expectations_fbl(Fs, Is, variabless, all_types, all_loops)

    prop_pos_specific_scc_of_loop = np.asarray(prop_pos_specific_scc_of_loop, dtype=float)

    exp_pos_null2 = np.nansum(nr_specific_k_loops * prop_pos_specific_scc_of_loop, axis=(0, 1))
    total_real_fbl = np.nansum(nr_specific_k_loops, axis=(0, 1))
    exp_neg_null2 = total_real_fbl - exp_pos_null2

    pos_over_exp = safe_div(obs_pos_total, exp_pos_null2)
    neg_over_exp = safe_div(obs_neg_total, exp_neg_null2)

    # ---- FFLs ----
    ffl_res = ad.compute_all_FFLs(Fs, Is, degrees, constantss, max_degree_used, N)
    nr_coherent = np.asarray(ffl_res[0], dtype=float)
    nr_incoherent = np.asarray(ffl_res[1], dtype=float)

    total_ffls = np.asarray(ffl_res[3], dtype=float).reshape(-1)
    prop_pos = np.asarray(prop_pos_separate2, dtype=float).reshape(-1)

    LEGEND = list(itertools.product([0, 1], repeat=3))
    npos = np.array([sum(bits) for bits in LEGEND], dtype=int)

    expected_per_type = np.vstack([
        total_ffls * (prop_pos ** npos[t]) * ((1.0 - prop_pos) ** (3 - npos[t]))
        for t in range(8)
    ])

    coherent_ids, incoherent_ids = [], []
    for t in range(8):
        signs = np.array([
            -1 if (t & 1) else  1,
            -1 if (t & 2) else  1,
            -1 if (t & 4) else  1
        ], dtype=int)
        (coherent_ids if can.is_ffl_coherent(signs) else incoherent_ids).append(t)

    exp_coh = np.nansum(expected_per_type[coherent_ids, :], axis=0)
    exp_incoh = np.nansum(expected_per_type[incoherent_ids, :], axis=0)

    coh_over_exp = safe_div(nr_coherent, exp_coh)
    incoh_over_exp = safe_div(nr_incoherent, exp_incoh)

    return {
        "FBL_pos_count": obs_pos_total,
        "FBL_neg_count": obs_neg_total,
        "FBL_pos_over_exp_null2": pos_over_exp,
        "FBL_neg_over_exp_null2": neg_over_exp,
        "FFL_coherent_count": nr_coherent,
        "FFL_incoherent_count": nr_incoherent,
        "FFL_coherent_over_exp": coh_over_exp,
        "FFL_incoherent_over_exp": incoh_over_exp,
    }
