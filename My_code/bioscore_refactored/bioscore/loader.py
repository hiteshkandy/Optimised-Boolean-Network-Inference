from __future__ import annotations
from typing import List, Tuple
import numpy as np

import analyse_database13 as ad
from bioscore.model import BooleanModel

def load_meta_analysis_models(max_degree: int = 12, max_N: int = 10_000) -> Tuple[List[BooleanModel], dict]:
    """Wraps ad.load_models_included_in_meta_analysis into a list of BooleanModel objects."""
    out = ad.load_models_included_in_meta_analysis(max_degree=max_degree, max_N=max_N)

    Fs                = out[0]
    Is                = out[1]
    degrees           = out[2]
    degrees_essential = out[3]
    variabless        = out[4]
    constantss        = out[5]
    models_loaded     = out[6]

    n_variables       = np.asarray(out[10], dtype=int)
    n_constants       = np.asarray(out[11], dtype=int)
    max_degree_used   = int(out[12])

    models: List[BooleanModel] = []
    for m, name in enumerate(models_loaded):
        nv = int(n_variables[m])
        nc = int(n_constants[m])

        F = []
        for f in Fs[m]:
            if f is None:
                F.append(None)
            else:
                F.append(np.asarray(f, dtype=int))

        I = [list(map(int, regs)) for regs in Is[m]]
        deg = np.asarray(degrees[m], dtype=int)
        deg_e = np.asarray(degrees_essential[m], dtype=int)

        variables = np.arange(nv, dtype=int)
        # constants are stored separately in ad outputs; we expose indices if contiguous
        constants = np.arange(nv, nv + nc, dtype=int) if (nv + nc) <= len(F) else np.array([], dtype=int)

        # pre-slice to genes (0..n_variables-1)
        F_genes = F[:nv]
        deg_genes = deg[:nv]
        deg_e_genes = deg_e[:nv]

        models.append(BooleanModel(
            name=str(name),
            F=F,
            I=I,
            degrees=deg,
            degrees_essential=deg_e,
            n_variables=nv,
            n_constants=nc,
            variables=variables,
            constants=constants,
            max_degree_used=max_degree_used,
            F_genes=F_genes,
            degrees_genes=deg_genes,
            degrees_essential_genes=deg_e_genes
        ))

    meta = {
        "models_loaded": models_loaded,
        "models_not_loaded": out[7],
        "similar_sets": out[8],
        "extra_bookkeeping": out[9],
        "max_degree_used": max_degree_used,
        "raw_out": out,  # keep for motif cache
        "variabless": variabless,
        "constantss": constantss,
        "n_variables": n_variables,
        "n_constants": n_constants,
    }
    return models, meta
