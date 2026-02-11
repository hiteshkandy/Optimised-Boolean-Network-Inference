from __future__ import annotations
import numpy as np

import canalizing_function_toolbox_v16 as can
import analyse_database13 as ad

from bioscore.model import BooleanModel, ScoreResult
from bioscore.config import ScoreConfig
from bioscore.validation import safe_nanmean, validate_truth_table_length

def _depths_per_node(model: BooleanModel) -> np.ndarray:
    nv = model.n_variables
    depths = np.full(nv, np.nan, dtype=float)
    for j in range(nv):
        f = model.F_genes[j]
        ke = int(model.degrees_essential_genes[j])
        if ke <= 0 or f is None:
            continue
        if not validate_truth_table_length(f, ke):
            continue
        try:
            (_, depth, _, _, _) = can.get_canalizing_depth_inputs_outputs_corefunction(np.asarray(f, dtype=int))
            depths[j] = float(depth)
        except Exception:
            depths[j] = np.nan
    return depths

def score_mean_canalizing_depth(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    depths = _depths_per_node(model)
    ke_arr = model.degrees_essential_genes.astype(float)
    mask = (ke_arr > 0) & (~np.isnan(depths))
    value = float(np.nanmean(depths[mask])) if mask.sum() else float("nan")
    return ScoreResult(
        name="mean_canalizing_depth",
        value=value,
        per_node=depths,
        meta={"n_nodes_used": int(mask.sum())}
    )

def score_fraction_nested_canalization(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    depths = _depths_per_node(model)
    ke_arr = model.degrees_essential_genes.astype(float)
    mask = (ke_arr > 0) & (~np.isnan(depths))
    frac = np.full_like(depths, np.nan, dtype=float)
    frac[mask] = depths[mask] / ke_arr[mask]
    value = float(np.nanmean(frac[mask])) if mask.sum() else float("nan")
    return ScoreResult(
        name="fraction_nested_canalization",
        value=value,
        per_node=frac,
        meta={"n_nodes_used": int(mask.sum())}
    )

def score_mean_canalizing_strength(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    Fs_essential = []
    for f, ke in zip(model.F_genes, model.degrees_essential_genes):
        d = int(ke)
        if d <= 0 or f is None:
            continue
        if not validate_truth_table_length(f, d):
            continue
        Fs_essential.append(np.asarray(f, dtype=int))

    if not Fs_essential:
        return ScoreResult(name="mean_canalizing_strength", value=float("nan"), meta={"n_functions": 0})

    cs = ad.get_canalizing_strengths([Fs_essential])[0]
    payload = cs[0] if (isinstance(cs, tuple) and len(cs) >= 1) else cs
    value = safe_nanmean(payload)
    return ScoreResult(name="mean_canalizing_strength", value=float(value), meta={"n_functions": len(Fs_essential)})
