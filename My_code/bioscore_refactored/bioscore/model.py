from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass(frozen=True)
class BooleanModel:
    """Canonical in-memory representation of one Boolean network model."""
    name: str
    F: List[Optional[np.ndarray]]          # truth tables per node (may include None)
    I: List[List[int]]                    # regulator indices per node
    degrees: np.ndarray                   # nominal in-degree per node (full model)
    degrees_essential: np.ndarray         # essential in-degree per node (full model)
    n_variables: int
    n_constants: int
    variables: np.ndarray                 # indices of variable nodes (typically 0..n_variables-1)
    constants: np.ndarray                 # indices of constant nodes
    max_degree_used: int

    # Pre-sliced "genes" view (variables only)
    F_genes: Optional[List[Optional[np.ndarray]]] = None
    degrees_genes: Optional[np.ndarray] = None
    degrees_essential_genes: Optional[np.ndarray] = None

@dataclass(frozen=True)
class ScoreResult:
    """Standard output of any score."""
    name: str
    value: float
    per_node: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None
