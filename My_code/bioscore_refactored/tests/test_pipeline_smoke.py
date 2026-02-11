from bioscore.config import ScoreConfig
from bioscore.loader import load_meta_analysis_models
from bioscore.pipeline import build_score_registry, compute_scores

def test_pipeline_smoke():
    cfg = ScoreConfig(max_degree=12, derrida_nsim=200)
    models, _ = load_meta_analysis_models(max_degree=cfg.max_degree, max_N=50)
    df = compute_scores(models, cfg, build_score_registry())
    assert df.shape[0] == len(models)
    assert "mean_canalizing_depth" in df.columns
    assert "derrida_value" in df.columns
