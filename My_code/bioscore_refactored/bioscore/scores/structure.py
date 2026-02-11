from __future__ import annotations
import numpy as np
from bioscore.model import BooleanModel, ScoreResult
from bioscore.config import ScoreConfig

def score_number_of_genes(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    return ScoreResult(name="n_genes", value=float(model.n_variables))

def score_number_of_constants(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    return ScoreResult(name="n_constants", value=float(model.n_constants))

def score_mean_in_degree(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    deg = model.degrees_genes.astype(float) if model.degrees_genes is not None else np.array([], dtype=float)
    return ScoreResult(name="mean_in_degree", value=float(np.mean(deg)) if deg.size else float("nan"))

def score_mean_essential_in_degree(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    ke = model.degrees_essential_genes.astype(float) if model.degrees_essential_genes is not None else np.array([], dtype=float)
    return ScoreResult(name="mean_essential_in_degree", value=float(np.mean(ke)) if ke.size else float("nan"))
