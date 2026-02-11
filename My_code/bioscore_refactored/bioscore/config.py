from dataclasses import dataclass

@dataclass(frozen=True)
class ScoreConfig:
    """Global configuration for score computation."""
    max_degree: int = 12
    derrida_nsim: int = 2000
    seed: int = 0
