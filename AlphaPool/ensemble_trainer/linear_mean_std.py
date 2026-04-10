"""Current ICIR-oriented linear ensemble trainer."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.correlation import max_abs_pearson
from .base import AbstractEnsembleTrainer, EnsembleFitResult
from .common import build_design_matrix, rolling_normalize_columns
from pool_state import PoolEntry


class LinearMeanStdEnsembleTrainer(AbstractEnsembleTrainer):
    """
    无约束权重线性 ensemble + L2 正则化。

    流程：
      1. 构建设计矩阵，按 combo_rolling_window 做 rolling z-score 归一化
         （window=0 时不归一化，直接使用原始值）
      2. 用 L-BFGS-B 在归一化空间内优化 -IC_score + l2 * ||w||^2
      3. 优化器与预测器使用完全相同的归一化矩阵，权重直接可用于最终预测

    与 zscore（全局静态归一化）相比，rolling 归一化是因果的（仅用历史数据），
    且优化目标与实际预测目标完全一致。
    """

    def __init__(
        self,
        maxiter: int = 400,
        l2_alpha: float = 1e-3,
        diversity_min_points: int = 10,
        combo_rolling_window: int = 20,
    ):
        self.maxiter = maxiter
        self.l2_alpha = l2_alpha
        self.diversity_min_points = diversity_min_points
        self.combo_rolling_window = combo_rolling_window

    def diversity_score(
        self,
        candidate: pd.Series,
        pool_factors: List[Any],
        data_idx: pd.Index,
    ) -> float:
        return float(
            max_abs_pearson(
                candidate,
                pool_factors,
                data_idx,
                min_points=self.diversity_min_points,
            )
        )

    def fit_and_predict(
        self,
        factors: List[PoolEntry],
        returns: pd.Series,
        data_idx: pd.Index,
        *,
        scorer,
    ) -> EnsembleFitResult:
        n = len(factors)
        if n == 0:
            return EnsembleFitResult(
                importance=np.zeros(0, dtype=np.float64),
                score=None,
                prediction=pd.Series(dtype=float),
            )

        # Align to the common index once; both optimizer and predictor use the
        # same X so the optimised weights are directly meaningful for prediction.
        common = data_idx.intersection(returns.index)
        X = build_design_matrix(factors, common)
        if self.combo_rolling_window > 0 and X.shape[1] > 0:
            X = rolling_normalize_columns(X, common, self.combo_rolling_window)

        l2 = self.l2_alpha

        def objective(w: np.ndarray) -> float:
            factor_s = pd.Series(X @ w, index=common)
            score_result = scorer.score_series(factor_s, returns, subset_index=data_idx)
            perf = float(score_result.score) if score_result.score is not None else -1e9
            return -perf + l2 * float(np.sum(w ** 2))

        w0 = np.ones(n, dtype=np.float64) / n
        res = minimize(objective, w0, method="L-BFGS-B", options={"maxiter": self.maxiter})
        w = res.x

        prediction = pd.Series(X @ w, index=common, name="prediction")
        score_result = scorer.score_series(prediction, returns, subset_index=data_idx)
        return EnsembleFitResult(
            importance=w,
            score=score_result.score,
            prediction=prediction,
        )


__all__ = ["LinearMeanStdEnsembleTrainer"]
