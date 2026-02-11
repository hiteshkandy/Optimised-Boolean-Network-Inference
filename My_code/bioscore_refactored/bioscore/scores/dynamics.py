from __future__ import annotations
import analyse_database13 as ad

from bioscore.model import BooleanModel, ScoreResult
from bioscore.config import ScoreConfig

def score_derrida_value(model: BooleanModel, cfg: ScoreConfig) -> ScoreResult:
    """Simulation-based Derrida value via ad.get_derrida_values (single-model wrapper)."""
    try:
        res = ad.get_derrida_values([model.F], [model.I], [model.degrees], max_degree=model.max_degree_used, nsim=cfg.derrida_nsim)
        derr = res[0][0] if (isinstance(res, tuple) and len(res) >= 1) else res[0]
        return ScoreResult(name="derrida_value", value=float(derr), meta={"nsim": cfg.derrida_nsim})
    except Exception:
        return ScoreResult(name="derrida_value", value=float("nan"), meta={"nsim": cfg.derrida_nsim, "error": "failed"})
