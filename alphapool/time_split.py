"""时间轴上划分训练 / 测试索引（可扩展为滚动月更等）。"""
from __future__ import annotations

import pandas as pd


def train_test_index_split(
    full_index: pd.Index,
    train_years: float = 2.0,
    test_fraction_if_no_test: float = 0.2,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    以时间顺序：从最早时刻起连续 train_years 年为训练段，其余为测试段。
    若数据不足两年或切分后测试段为空，则退化为按比例留出 test_fraction_if_no_test 作为测试集。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(full_index).unique())).sort_values()
    if len(idx) < 10:
        split = max(1, int(len(idx) * 0.8))
        return idx[:split], idx[split:]

    t0 = idx[0]
    cut = t0 + pd.DateOffset(years=float(train_years))
    train_idx = idx[idx <= cut]
    test_idx = idx[idx > cut]

    if len(train_idx) == 0:
        split = max(1, int(len(idx) * (1.0 - test_fraction_if_no_test)))
        train_idx = idx[:split]
        test_idx = idx[split:]
    elif len(test_idx) == 0:
        split = max(1, int(len(idx) * (1.0 - test_fraction_if_no_test)))
        train_idx = idx[:split]
        test_idx = idx[split:]

    return train_idx, test_idx
