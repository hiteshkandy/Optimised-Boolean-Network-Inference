from __future__ import annotations
import numpy as np
import canalizing_function_toolbox_v16 as can

from bioscore.model import BooleanModel, ScoreResult
from bioscore.config import ScoreConfig
from bioscore.validation import safe_nanmean, validate_truth_table_length

def _node_bias_p1_minus_p(f: np.ndarray) -> float:
    p1 = float(np.sum(f)) / float(len(f))
    return p1 * (1.0 - p1)

def score_mean_bias_p1_minus_p(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    vals = []
    for f, ke in zip(model.F_genes, model.degrees_essential_genes):
        d = int(ke)
        if d <= 0 or f is None:
            continue
        if not validate_truth_table_length(f, d):
            continue
        vals.append(_node_bias_p1_minus_p(np.asarray(f, dtype=int)))
    return ScoreResult(name="mean_bias_p1_minus_p", value=float(safe_nanmean(vals)), meta={"n_nodes": len(vals)})

def score_mean_input_redundancy(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    vals = []
    for f, ke in zip(model.F_genes, model.degrees_essential_genes):
        d = int(ke)
        if d <= 0 or f is None:
            continue
        if not validate_truth_table_length(f, d):
            continue
        if d > 8:
            vals.append(np.nan); continue
        try:
            vals.append(can.get_input_redundancy(np.asarray(f, dtype=int), n=d))
        except Exception:
            vals.append(np.nan)
    return ScoreResult(name="mean_input_redundancy", value=float(safe_nanmean(vals)), meta={"n_nodes": len(vals)})

def score_mean_effective_connectivity(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    vals = []
    for f, ke in zip(model.F_genes, model.degrees_essential_genes):
        d = int(ke)
        if d <= 0 or f is None:
            continue
        if not validate_truth_table_length(f, d):
            continue
        try:
            effs = can.get_edge_effectiveness(np.asarray(f, dtype=int), n=d)
            vals.append(float(np.nansum(effs)))
        except Exception:
            vals.append(np.nan)
    return ScoreResult(name="mean_effective_connectivity", value=float(safe_nanmean(vals)), meta={"n_nodes": len(vals)})

def score_covariance_Ke_and_bias(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    ks = []
    bs = []
    for f, ke in zip(model.F_genes, model.degrees_essential_genes):
        d = int(ke)
        if d <= 0 or f is None:
            continue
        if not validate_truth_table_length(f, d):
            continue
        ks.append(float(d))
        bs.append(_node_bias_p1_minus_p(np.asarray(f, dtype=int)))

    if len(ks) < 2:
        return ScoreResult(name="cov_Ke_bias", value=float("nan"), meta={"n_nodes": len(ks)})

    k = np.asarray(ks, dtype=float)
    b = np.asarray(bs, dtype=float)
    cov = float(np.cov(k, b, bias=True)[0, 1])
    return ScoreResult(name="cov_Ke_bias", value=cov, meta={"n_nodes": len(ks)})

def score_Ke_times_p1_minus_p(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    ke = model.degrees_essential_genes.astype(float)
    ke_mean = float(np.nanmean(ke)) if ke.size else float("nan")
    bias = score_mean_bias_p1_minus_p(model, cfg).value
    return ScoreResult(name="Ke_times_p1_minus_p", value=float(ke_mean) * float(bias))

def score_Ke_times_p1_minus_p_plus_cov(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    base = score_Ke_times_p1_minus_p(model, cfg).value
    covv = score_covariance_Ke_and_bias(model, cfg).value
    return ScoreResult(name="Ke_times_p1_minus_p_plus_cov", value=float(base + covv))

def score_K_times_p1_minus_p(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    k = model.degrees_genes.astype(float)
    k_mean = float(np.nanmean(k)) if k.size else float("nan")
    bias = score_mean_bias_p1_minus_p(model, cfg).value
    return ScoreResult(name="K_times_p1_minus_p", value=float(k_mean) * float(bias))
