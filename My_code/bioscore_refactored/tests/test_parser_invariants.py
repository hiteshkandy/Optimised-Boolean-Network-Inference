import numpy as np
from bioscore.loader import load_meta_analysis_models

def _tt_len(f):
    """
    f can be a list/np.ndarray of 0/1 values, or something nested.
    We only support 1D truth tables here; nested truth tables should be flattened upstream.
    """
    if f is None:
        return None
    arr = np.asarray(f)
    if arr.ndim != 1:
        # if you *do* store nested (e.g., 2D), treat total size as length
        return int(arr.size)
    return int(arr.shape[0])

def test_degrees_and_truth_table_lengths_are_consistent():
    models, _ = load_meta_analysis_models(max_degree=12, max_N=200)
    assert len(models) > 0

    for m in models[:20]:
        # per-gene degree arrays must align
        assert len(m.F_genes) == m.n_variables
        assert m.degrees_genes.shape[0] == m.n_variables
        assert m.degrees_essential_genes.shape[0] == m.n_variables

        for f, k, ke in zip(m.F_genes, m.degrees_genes, m.degrees_essential_genes):
            k = int(k)
            ke = int(ke)

            # 1) essential degree should never exceed (listed) degree
            assert ke <= k, f"ke={ke} > k={k} for a gene in model (N={m.n_variables})"

            # nothing further to check without a truth table or with degenerate ke/k
            if f is None:
                continue

            L = _tt_len(f)
            if L is None or L <= 0:
                continue

            # 2) truth table length must be either 2^ke (reduced TT) OR 2^k (full TT)
            expected_reduced = 1 << ke  # 2**ke
            expected_full    = 1 << k   # 2**k

            assert L in (expected_reduced, expected_full), (
                f"Truth table length mismatch: len={L}, expected 2^ke={expected_reduced} or 2^k={expected_full} "
                f"(k={k}, ke={ke}, N={m.n_variables})"
            )
