"""
线性因子池权重训练（MeanStd 风格：在全体数据上优化目标）。
未来可继承 AbstractEnsembleTrainer 换为 LGBM + importance。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from helpers.reward_metrics import performance_score_on_subset

from scipy.optimize import minimize


def _softmax(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    u = u - np.max(u)
    e = np.exp(np.clip(u, -30, 30))
    return e / (e.sum() + 1e-12)


def _build_design_matrix(factors: List[pd.Series], index: pd.Index) -> np.ndarray:
    cols = []
    for s in factors:
        v = s.reindex(index).values.astype(np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(v)
    if not cols:
        return np.zeros((len(index), 0))
    return np.column_stack(cols)


@dataclass
class EnsembleFitResult:
    weights: np.ndarray
    score: Optional[float]


def build_full_prediction(
    factors: List[pd.Series],
    returns: pd.Series,
    weights: np.ndarray,
) -> pd.Series:
    """线性组合：在全体因子与 returns 的公共索引上生成预测序列。"""
    if not factors:
        return pd.Series(dtype=float)
    common = factors[0].index
    for f in factors[1:]:
        common = common.intersection(f.index)
    common = common.intersection(returns.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != len(factors):
        raise ValueError(f"weights length {w.shape[0]} != num factors {len(factors)}")
    X = _build_design_matrix(factors, common)
    pred = X @ w
    return pd.Series(pred, index=common, name="prediction")


class AbstractEnsembleTrainer(ABC):
    """可扩展：例如 LGBMEnsembleTrainer。"""

    @abstractmethod
    def gfit(
        self,
        factors: List[pd.Series],
        returns: pd.Series,
        data_idx: pd.Index,
        *,
        icir_missing_gamma: float,
        icir_missing_eps: float,
    ) -> EnsembleFitResult:
        ...


class LinearMeanStdEnsembleTrainer(AbstractEnsembleTrainer):
    """
    在全体数据上用 softmax 参数化权重，最小化「负的 performance_score」（与 eval_final_reward 一致）。
    """

    def __init__(self, maxiter: int = 400):
        self.maxiter = maxiter

    def gfit(
        self,
        factors: List[pd.Series],
        returns: pd.Series,
        data_idx: pd.Index,
        *,
        icir_missing_gamma: float,
        icir_missing_eps: float,
    ) -> EnsembleFitResult:
        n = len(factors)
        X = _build_design_matrix(factors, data_idx)
        r = returns.reindex(data_idx).values.astype(np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

        def objective(u: np.ndarray) -> float:
            w = _softmax(u)
            pred = X @ w
            factor_s = pd.Series(pred, index=data_idx)
            ret_s = pd.Series(r, index=data_idx)
            sc = performance_score_on_subset(
                factor_s,
                ret_s,
                data_idx,
                icir_missing_gamma=icir_missing_gamma,
                icir_missing_eps=icir_missing_eps,
                eval_mode="full_weighted",
            )
            return -float(sc) if sc is not None else 1e9

        u0 = np.zeros(n)
        res = minimize(objective, u0, method="L-BFGS-B", options={"maxiter": self.maxiter})
        w_opt = _softmax(res.x)
        return self._pack_result(
            w_opt, factors, returns, data_idx,
            icir_missing_gamma, icir_missing_eps,
        )

    def _pack_result(
        self,
        w: np.ndarray,
        factors: List[pd.Series],
        returns: pd.Series,
        data_idx: pd.Index,
        icir_missing_gamma: float,
        icir_missing_eps: float,
    ) -> EnsembleFitResult:
        X = _build_design_matrix(factors, data_idx)
        r = returns.reindex(data_idx).values.astype(np.float64)
        r = np.nan_to_num(r, nan=0.0)
        pred = X @ w
        s = pd.Series(pred, index=data_idx)
        sc = performance_score_on_subset(
            s, pd.Series(r, index=data_idx), data_idx,
            icir_missing_gamma=icir_missing_gamma,
            icir_missing_eps=icir_missing_eps,
            eval_mode="full_weighted",
        )
        return EnsembleFitResult(weights=w, score=sc)
