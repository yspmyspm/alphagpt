"""池内因子与候选因子之间的相关性。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def max_abs_pearson(
    candidate: pd.Series,
    pool_factors: List[pd.Series],
    index: pd.Index,
    min_points: int = 10,
) -> float:
    """
    在给定 index 上，计算 candidate 与 pool 中每个因子 Pearson 相关系数绝对值的最大值。
    pool 为空时返回 0。
    """
    if not pool_factors:
        return 0.0
    cx = candidate.reindex(index).values.astype(np.float64)
    best = 0.0
    for p in pool_factors:
        px = p.reindex(index).values.astype(np.float64)
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


def pairwise_corr_matrix(factors: List[pd.Series], index: pd.Index) -> Optional[np.ndarray]:
    """因子间 |corr| 矩阵 (n, n)，用于调试。"""
    if not factors:
        return None
    n = len(factors)
    X = np.column_stack(
        [np.nan_to_num(f.reindex(index).values.astype(np.float64), nan=0.0) for f in factors]
    )
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            if m.sum() < 5:
                continue
            c = np.corrcoef(X[m, i], X[m, j])[0, 1]
            if np.isfinite(c):
                out[i, j] = out[j, i] = abs(c)
    return out
