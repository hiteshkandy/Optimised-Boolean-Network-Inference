from __future__ import annotations
from typing import List
import pandas as pd

from bioscore.model import BooleanModel
from bioscore.config import ScoreConfig
from bioscore.scores.base import ScoreSpec

from bioscore.scores.structure import (
    score_number_of_genes, score_number_of_constants, score_mean_in_degree, score_mean_essential_in_degree
)
from bioscore.scores.canalization import (
    score_mean_canalizing_depth, score_fraction_nested_canalization, score_mean_canalizing_strength
)
from bioscore.scores.canalization_extra import (
    score_mean_bias_p1_minus_p, score_mean_input_redundancy, score_mean_effective_connectivity,
    score_covariance_Ke_and_bias, score_Ke_times_p1_minus_p, score_Ke_times_p1_minus_p_plus_cov,
    score_K_times_p1_minus_p
)
from bioscore.scores.dynamics import score_derrida_value

def build_score_registry() -> List[ScoreSpec]:
    """Central list of (paper-native) score functions used to build feature tables."""
    return [
        # structural
        ScoreSpec("n_genes", score_number_of_genes),
        ScoreSpec("n_constants", score_number_of_constants),
        ScoreSpec("mean_in_degree", score_mean_in_degree),
        ScoreSpec("mean_essential_in_degree", score_mean_essential_in_degree),

        # canalization
        ScoreSpec("mean_canalizing_depth", score_mean_canalizing_depth),
        ScoreSpec("fraction_nested_canalization", score_fraction_nested_canalization),
        ScoreSpec("mean_canalizing_strength", score_mean_canalizing_strength),

        # bias/redundancy/effective connectivity + composites
        ScoreSpec("mean_bias_p1_minus_p", score_mean_bias_p1_minus_p),
        ScoreSpec("mean_input_redundancy", score_mean_input_redundancy),
        ScoreSpec("mean_effective_connectivity", score_mean_effective_connectivity),
        ScoreSpec("cov_Ke_bias", score_covariance_Ke_and_bias),
        ScoreSpec("Ke_times_p1_minus_p", score_Ke_times_p1_minus_p),
        ScoreSpec("Ke_times_p1_minus_p_plus_cov", score_Ke_times_p1_minus_p_plus_cov),
        ScoreSpec("K_times_p1_minus_p", score_K_times_p1_minus_p),

        # dynamics
        ScoreSpec("derrida_value", score_derrida_value),
    ]

def compute_scores(models: List[BooleanModel], cfg: ScoreConfig, registry: List[ScoreSpec]) -> pd.DataFrame:
    rows = []
    for m in models:
        row = {"model_name": m.name}
        for spec in registry:
            res = spec.fn(m, cfg)
            row[spec.name] = res.value
        rows.append(row)
    return pd.DataFrame(rows)
