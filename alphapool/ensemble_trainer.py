"""
线性因子池权重训练（MeanStd 风格：在训练段优化目标，在测试段用 eval_final_reward 打分）。
未来可继承 AbstractEnsembleTrainer 换为 LGBM + importance。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from helpers.reward_metrics import performance_score_on_subset

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


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
    train_score: Optional[float]
    test_score: Optional[float]


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
    def fit(
        self,
        factors: List[pd.Series],
        returns: pd.Series,
        train_idx: pd.Index,
        test_idx: pd.Index,
        *,
        icir_missing_gamma: float,
        icir_missing_eps: float,
        fragment_eval: bool = True,
    ) -> EnsembleFitResult:
        ...


class LinearMeanStdEnsembleTrainer(AbstractEnsembleTrainer):
    """
    在训练段用 softmax 参数化权重，最小化「负的 performance_score」（与 eval_final_reward 一致）。
    测试段用同样指标评估组合预测。
    """

    def __init__(self, maxiter: int = 400):
        self.maxiter = maxiter

    def fit(
        self,
        factors: List[pd.Series],
        returns: pd.Series,
        train_idx: pd.Index,
        test_idx: pd.Index,
        *,
        icir_missing_gamma: float,
        icir_missing_eps: float,
        fragment_eval: bool = True,
    ) -> EnsembleFitResult:
        if not factors or len(train_idx) == 0 or len(test_idx) == 0:
            return EnsembleFitResult(
                weights=np.zeros(len(factors)),
                train_score=None,
                test_score=None,
            )

        n = len(factors)
        X_tr = _build_design_matrix(factors, train_idx)
        r_tr = returns.reindex(train_idx).values.astype(np.float64)
        r_tr = np.nan_to_num(r_tr, nan=0.0, posinf=0.0, neginf=0.0)

        ev_mode = "fragment_ic" if fragment_eval else "full_weighted"

        if minimize is None:
            w = np.ones(n) / n
            return self._pack_result(
                w, factors, returns, train_idx, test_idx,
                icir_missing_gamma, icir_missing_eps, ev_mode,
            )

        def objective(u: np.ndarray) -> float:
            w = _softmax(u)
            pred = X_tr @ w
            factor_s = pd.Series(pred, index=train_idx)
            ret_s = pd.Series(r_tr, index=train_idx)
            sc = performance_score_on_subset(
                factor_s,
                ret_s,
                train_idx,
                icir_missing_gamma=icir_missing_gamma,
                icir_missing_eps=icir_missing_eps,
                eval_mode=ev_mode,
            )
            return -float(sc) if sc is not None else 1e9

        u0 = np.zeros(n)
        res = minimize(objective, u0, method="L-BFGS-B", options={"maxiter": self.maxiter})
        w_opt = _softmax(res.x) if res.success else np.ones(n) / n
        return self._pack_result(
            w_opt, factors, returns, train_idx, test_idx,
            icir_missing_gamma, icir_missing_eps, ev_mode,
        )

    def _pack_result(
        self,
        w: np.ndarray,
        factors: List[pd.Series],
        returns: pd.Series,
        train_idx: pd.Index,
        test_idx: pd.Index,
        icir_missing_gamma: float,
        icir_missing_eps: float,
        eval_mode: str,
    ) -> EnsembleFitResult:
        X_tr = _build_design_matrix(factors, train_idx)
        X_te = _build_design_matrix(factors, test_idx)
        r_tr = returns.reindex(train_idx).values.astype(np.float64)
        r_te = returns.reindex(test_idx).values.astype(np.float64)
        r_tr = np.nan_to_num(r_tr, nan=0.0)
        r_te = np.nan_to_num(r_te, nan=0.0)

        pred_tr = X_tr @ w
        pred_te = X_te @ w
        tr_s = pd.Series(pred_tr, index=train_idx)
        te_s = pd.Series(pred_te, index=test_idx)
        train_sc = performance_score_on_subset(
            tr_s, pd.Series(r_tr, index=train_idx), train_idx,
            icir_missing_gamma=icir_missing_gamma,
            icir_missing_eps=icir_missing_eps,
            eval_mode=eval_mode,
        )
        test_sc = performance_score_on_subset(
            te_s, pd.Series(r_te, index=test_idx), test_idx,
            icir_missing_gamma=icir_missing_gamma,
            icir_missing_eps=icir_missing_eps,
            eval_mode=eval_mode,
        )
        return EnsembleFitResult(weights=w, train_score=train_sc, test_score=test_sc)
