"""
SeriesOps：唯一算子实现，VM 通过遍历此类获取算子。
特征计算在 pd.Series 上完成，无 torch。
"""
import inspect
import numpy as np
import pandas as pd

from helpers.ops_core import OpInner
from config import ModelConfig


class SeriesOps:
    """
    算子类，VM 通过 dir(SeriesOps) 遍历获取，不依赖 config 列表。
    """

    @staticmethod
    def add(x: pd.Series, y: pd.Series) -> pd.Series:
        xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        return pd.Series(OpInner.add(xv, yv), index=x.index)

    @staticmethod
    def sub(x: pd.Series, y: pd.Series) -> pd.Series:
        xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        return pd.Series(OpInner.sub(xv, yv), index=x.index)

    @staticmethod
    def mul(x: pd.Series, y: pd.Series) -> pd.Series:
        xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        return pd.Series(OpInner.mul(xv, yv), index=x.index)

    @staticmethod
    def div(x: pd.Series, y: pd.Series) -> pd.Series:
        xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        return pd.Series(OpInner.div(xv, yv), index=x.index)

    @staticmethod
    def neg(x: pd.Series) -> pd.Series:
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.neg(xv), index=x.index)

    @staticmethod
    def abs_(x: pd.Series) -> pd.Series:
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.abs_(xv), index=x.index)

    @staticmethod
    def sign(x: pd.Series) -> pd.Series:
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.sign(xv), index=x.index)

    @staticmethod
    def delay(x: pd.Series) -> pd.Series:
        """使用 config 中注册的 DELAY_PERIOD"""
        d = ModelConfig.DELAY_PERIOD
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.delay(xv, d), index=x.index)

    @staticmethod
    def gate(cond: pd.Series, x: pd.Series, y: pd.Series) -> pd.Series:
        cv = cond.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        return pd.Series(OpInner.gate(cv, xv, yv), index=x.index)

    @staticmethod
    def jump(x: pd.Series) -> pd.Series:
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.jump(xv), index=x.index)

    @staticmethod
    def decay(x: pd.Series) -> pd.Series:
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.decay(xv), index=x.index)

    @staticmethod
    def delay1(x: pd.Series) -> pd.Series:
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.delay1(xv), index=x.index)

    @staticmethod
    def max3(x: pd.Series) -> pd.Series:
        xv = x.to_numpy(dtype=float)
        return pd.Series(OpInner.max3(xv), index=x.index)


def _collect_ops():
    """遍历 SeriesOps 类，收集 (name, func, arity)。"""
    out = []
    for name in dir(SeriesOps):
        if name.startswith("_"):
            continue
        attr = getattr(SeriesOps, name)
        if not callable(attr):
            continue
        sig = inspect.signature(attr)
        arity = len(sig.parameters)
        out.append((name, attr, arity))
    return out


def get_ops():
    """返回 [(name, func, arity), ...]，供 VM 使用。"""
    return _collect_ops()
