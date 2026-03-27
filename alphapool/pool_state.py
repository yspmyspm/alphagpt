"""Alpha pool：持久化表达式与因子时间序列（可扩展落盘）。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class PoolEntry:
    formula: List[int]
    factor: pd.Series


class AlphaPoolState:
    """容量上限为 capacity；条目为 (token 公式, 因子序列)。"""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.entries: List[PoolEntry] = []

    def factor_series_list(self) -> List[pd.Series]:
        return [e.factor for e in self.entries]

    def clear(self) -> None:
        self.entries.clear()

    def replace_entries(self, new_entries: List[PoolEntry]) -> None:
        self.entries = new_entries[: self.capacity]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    @classmethod
    def load(cls, path: str, capacity: int) -> "AlphaPoolState":
        p = cls(capacity)
        with open(path, "rb") as f:
            p.entries = pickle.load(f)
        p.entries = p.entries[:capacity]
        return p
