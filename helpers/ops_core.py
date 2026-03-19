"""
内层计算：纯 numpy，无 for loop / rolling.apply。
供 SeriesOps 调用。
"""
import numpy as np


def _delay(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        return x
    out = np.empty_like(x)
    out[:d] = np.nan
    out[d:] = x[:-d]
    return out


class OpInner:
    """numpy 计算层，向量化实现"""

    @staticmethod
    def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    @staticmethod
    def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - y

    @staticmethod
    def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y

    @staticmethod
    def div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x / (np.abs(y) + 1e-6)

    @staticmethod
    def neg(x: np.ndarray) -> np.ndarray:
        return -x

    @staticmethod
    def abs_(x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    @staticmethod
    def sign(x: np.ndarray) -> np.ndarray:
        return np.sign(x)

    @staticmethod
    def delay(x: np.ndarray, d: int) -> np.ndarray:
        return _delay(x, d)

    @staticmethod
    def gate(cond: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mask = (cond > 0).astype(np.float64)
        return mask * x + (1.0 - mask) * y

    @staticmethod
    def jump(x: np.ndarray) -> np.ndarray:
        mean = np.nanmean(x)
        std = np.nanstd(x) + 1e-6
        z = (x - mean) / std
        return np.maximum(z - 3.0, 0.0)

    @staticmethod
    def decay(x: np.ndarray) -> np.ndarray:
        return x + 0.8 * _delay(x, 1) + 0.6 * _delay(x, 2)

    @staticmethod
    def delay1(x: np.ndarray) -> np.ndarray:
        return _delay(x, 1)

    @staticmethod
    def max3(x: np.ndarray) -> np.ndarray:
        d1 = _delay(x, 1)
        d2 = _delay(x, 2)
        return np.maximum(np.maximum(x, d1), d2)
