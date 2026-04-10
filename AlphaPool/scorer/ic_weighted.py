"""Current IC/ICIR-based scorer."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.metrics import compute_ic_metrics, compute_ic_score_details
from .base import AbstractSeriesScorer, ScoreResult


class ICWeightedScorer(AbstractSeriesScorer):
    """
    Combine daily ICIR, monthly ICIR, and a normalized IC term into one score.
    """

    def __init__(
        self,
        icir_missing_gamma: float = 2.0,
        icir_missing_eps: float = 1e-6,
        w_daily: float = 0.3,
        w_monthly: float = 0.3,
        w_ic: float = 0.4,
    ):
        self.icir_missing_gamma = float(icir_missing_gamma)
        self.icir_missing_eps = float(icir_missing_eps)
        self.w_daily = float(w_daily)
        self.w_monthly = float(w_monthly)
        self.w_ic = float(w_ic)

    def score_series(
        self,
        series: pd.Series,
        returns: pd.Series,
        *,
        subset_index: Optional[pd.Index] = None,
    ) -> ScoreResult:
        ic = compute_ic_metrics(series, returns, subset_index=subset_index)
        if ic is None:
            return ScoreResult(score=None, metrics={})

        daily_ic, monthly_ic, overall_ic = ic
        if not np.isfinite(overall_ic):
            return ScoreResult(score=None, metrics={})

        detail = compute_ic_score_details(
            daily_ic,
            monthly_ic,
            overall_ic,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
            w_daily=self.w_daily,
            w_monthly=self.w_monthly,
            w_ic=self.w_ic,
        )
        score = float(detail["combined_score"])
        if not np.isfinite(score):
            return ScoreResult(score=None, metrics={})

        metrics = dict(detail)
        metrics.pop("combined_score", None)
        return ScoreResult(score=float(score), metrics=metrics)


__all__ = ["ICWeightedScorer"]
