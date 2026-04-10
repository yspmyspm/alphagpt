"""池内因子与候选因子之间的相关性。"""
from __future__ import annotations

from typing import Any, List, Optional

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


def max_abs_pearson(
    candidate: Any,
    pool_factors: List[Any],
    index: pd.Index,
    min_points: int = 10,
    *,
    candidate_values: Optional[np.ndarray] = None,
    pool_dense_values: Optional[List[np.ndarray]] = None,
) -> float:
    """
    在给定 index 上，计算 candidate 与 pool 中每个因子 Pearson 相关系数绝对值的最大值。
    pool 为空时返回 0。
    """
    if not pool_factors:
        return 0.0
    cx = (
        np.asarray(candidate_values, dtype=np.float64)
        if candidate_values is not None
        else _dense_values(candidate, index)
    )
    best = 0.0
    for idx, p in enumerate(pool_factors):
        if pool_dense_values is not None and idx < len(pool_dense_values):
            px = np.asarray(pool_dense_values[idx], dtype=np.float64)
        else:
            px = _dense_values(p, index)
        m = np.isfinite(cx) & np.isfinite(px)
        if m.sum() < min_points:
            continue
        a, b = cx[m], px[m]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        cc = np.corrcoef(a, b)[0, 1]
        if np.isfinite(cc):
            best = max(best, abs(float(cc)))
    return float(best)
