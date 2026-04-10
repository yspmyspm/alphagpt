"""Shared matrix-prep helpers for trainers."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd


def _dense_values(item: Any, index: pd.Index) -> np.ndarray:
    if isinstance(item, np.ndarray):
        out = np.asarray(item, dtype=np.float64)
        if out.shape[0] != len(index):
            raise ValueError(f"dense array length {out.shape[0]} != index length {len(index)}")
        return out
    dense_values = getattr(item, "dense_values", None)
    if dense_values is not None:
        out = np.asarray(dense_values, dtype=np.float64)
        if out.shape[0] == len(index):
            return out
    if isinstance(item, pd.Series):
        return item.reindex(index).to_numpy(dtype=np.float64, copy=False)
    factor = getattr(item, "factor", None)
    if isinstance(factor, pd.Series):
        return factor.reindex(index).to_numpy(dtype=np.float64, copy=False)
    raise TypeError(f"unsupported factor type: {type(item).__name__}")


def build_design_matrix(factors: List[Any], index: pd.Index) -> np.ndarray:
    cols = []
    for s in factors:
        v = _dense_values(s, index)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(v)
    if not cols:
        return np.zeros((len(index), 0))
    return np.column_stack(cols)


def rolling_normalize_columns(X: np.ndarray, index: pd.Index, window: int) -> np.ndarray:
    """Apply rolling z-score normalization (min_periods=1) to all columns at once."""
    df = pd.DataFrame(X, index=index, dtype=np.float64)
    roll = df.rolling(window=window, min_periods=1)
    roll_std = roll.std().where(lambda s: s.notna() & (s > 0), other=1.0)
    return ((df - roll.mean()) / roll_std).fillna(0.0).to_numpy(dtype=np.float64)


__all__ = [
    "build_design_matrix",
    "rolling_normalize_columns",
]
