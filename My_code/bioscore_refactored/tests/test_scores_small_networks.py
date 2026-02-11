import numpy as np
from bioscore.model import BooleanModel
from bioscore.config import ScoreConfig
from bioscore.scores.canalization_extra import score_mean_bias_p1_minus_p

def make_toy_model_AND():
    # AND truth table under standard ordering: [0,0,0,1]
    f = np.array([0, 0, 0, 1], dtype=int)
    return BooleanModel(
        name="toy_AND",
        F=[f],
        I=[[0, 0]],
        degrees=np.array([2], dtype=int),
        degrees_essential=np.array([2], dtype=int),
        n_variables=1,
        n_constants=0,
        variables=np.array([0], dtype=int),
        constants=np.array([], dtype=int),
        max_degree_used=12,
        F_genes=[f],
        degrees_genes=np.array([2], dtype=int),
        degrees_essential_genes=np.array([2], dtype=int),
    )

def test_bias_AND():
    cfg = ScoreConfig()
    m = make_toy_model_AND()
    res = score_mean_bias_p1_minus_p(m, cfg)
    # p=0.25 -> p(1-p)=0.1875
    assert abs(res.value - 0.1875) < 1e-9
