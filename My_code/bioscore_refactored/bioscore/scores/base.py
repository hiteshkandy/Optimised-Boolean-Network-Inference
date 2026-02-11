from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from bioscore.model import BooleanModel, ScoreResult
from bioscore.config import ScoreConfig

ScoreFn = Callable[[BooleanModel, ScoreConfig], ScoreResult]

@dataclass(frozen=True)
class ScoreSpec:
    name: str
    fn: ScoreFn
