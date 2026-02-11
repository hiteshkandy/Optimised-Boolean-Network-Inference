from __future__ import annotations
import numpy as np

def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = (b != 0) & ~np.isnan(a) & ~np.isnan(b)
    out[m] = a[m] / b[m]
    return out

def safe_nanmean(x) -> float:
    """Robust nanmean from scalars/lists/arrays/object arrays."""
    vals = []
    def _add(item):
        if item is None:
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            vals.append(float(item)); return
        if isinstance(item, np.ndarray):
            if item.dtype == object:
                for sub in item.ravel():
                    _add(sub)
            else:
                try:
                    vals.extend(item.astype(float, copy=False).ravel().tolist())
                except Exception:
                    for sub in item.ravel():
                        _add(sub)
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                _add(sub)
            return
        try:
            vals.append(float(item))
        except Exception:
            return

    _add(x)
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")

def validate_truth_table_length(f: np.ndarray, k: int) -> bool:
    return (f is not None) and (len(f) == (2 ** int(k)))
