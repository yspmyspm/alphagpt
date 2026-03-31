"""
单轮 batch：并行得到 factor 后，按 alphapool 规则写 reward，
所有合规候选联合训练后裁剪 pool。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from alphapool.correlation import max_abs_pearson
from alphapool.ensemble_trainer import (
    AbstractEnsembleTrainer,
    LinearMeanStdEnsembleTrainer,
    build_full_prediction,
)
from alphapool.pool_feature_logging import save_removed_pool_features
from alphapool.pool_state import AlphaPoolState, PoolEntry
from configs import ModelConfig
from helpers.reward_metrics import compute_ic_metrics


@dataclass
class PoolMetrics:
    score: Optional[float]
    overall_ic: float
    daily_icir: float
    monthly_icir: float
    daily_coverage: float
    monthly_coverage: float


@dataclass
class BatchSummary:
    pool_size_before: int
    pool_size_after: int
    pool_before: Optional[PoolMetrics]
    pool_after: Optional[PoolMetrics]
    n_candidates: int


class AlphaPoolBatchEvaluator:
    def __init__(
        self,
        returns: pd.Series,
        data_idx: pd.Index,
        pool: AlphaPoolState,
        trainer: AbstractEnsembleTrainer,
        *,
        icir_missing_gamma: float,
        icir_missing_eps: float,
        missing_threshold: float,
        execute_fail_penalty: float,
        missing_high_penalty: float,
        low_std_penalty_base: float,
        compliance_fail_penalty: float,
        max_corr_min_points: int,
    ):
        self.returns = returns
        self.data_idx = data_idx
        self.pool = pool
        self.trainer = trainer
        self.icir_missing_gamma = icir_missing_gamma
        self.icir_missing_eps = icir_missing_eps
        self.missing_threshold = missing_threshold
        self.execute_fail_penalty = execute_fail_penalty
        self.missing_high_penalty = missing_high_penalty
        self.low_std_penalty_base = low_std_penalty_base
        self.compliance_fail_penalty = compliance_fail_penalty
        self.max_corr_min_points = int(max_corr_min_points)
        self._cached_pool_metrics: Optional[PoolMetrics] = None

    def clear_cache(self) -> None:
        self._cached_pool_metrics = None

    @classmethod
    def default_trainer(cls) -> LinearMeanStdEnsembleTrainer:
        return LinearMeanStdEnsembleTrainer(
            maxiter=int(ModelConfig.ENSEMBLE_MAXITER),
        )

    def _pool_metrics(self) -> Optional[PoolMetrics]:
        factors = self.pool.factor_series_list()
        if not factors:
            return None
        fit = self.trainer.gfit(
            factors,
            self.returns,
            self.data_idx,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
        )
        pred = build_full_prediction(
            factors,
            self.returns,
            fit.weights,
        )
        zero = PoolMetrics(score=fit.score, overall_ic=0, daily_icir=0, monthly_icir=0,
                           daily_coverage=0, monthly_coverage=0)
        if len(pred) == 0:
            return zero
        ic = compute_ic_metrics(pred, self.returns)
        if ic is None:
            return zero
        daily_ic, monthly_ic, overall_ic = ic
        daily_std = daily_ic.std()
        monthly_std = monthly_ic.std()
        daily_icir = float(daily_ic.mean() / daily_std) if daily_std > 1e-8 else 0.0
        monthly_icir = float(monthly_ic.mean() / monthly_std) if monthly_std > 1e-8 else 0.0
        daily_cov = float(np.isfinite(daily_ic.to_numpy(dtype=np.float64)).mean()) if len(daily_ic) > 0 else 0.0
        monthly_cov = float(np.isfinite(monthly_ic.to_numpy(dtype=np.float64)).mean()) if len(monthly_ic) > 0 else 0.0
        return PoolMetrics(
            score=fit.score,
            overall_ic=float(overall_ic),
            daily_icir=daily_icir,
            monthly_icir=monthly_icir,
            daily_coverage=daily_cov,
            monthly_coverage=monthly_cov,
        )

    def _reward_single(self, r: Dict[str, Any], pool_factors: List[pd.Series]) -> float:
        if not r.get("ok", False):
            return float(self.execute_fail_penalty)
        if r.get("missing_high"):
            return float(self.missing_high_penalty)
        if r.get("low_std"):
            return float(r.get("reward", self.low_std_penalty_base))

        factor: pd.Series = r["factor"]
        compliance = float(r.get("compliance", 0.0))
        max_c = max_abs_pearson(
            factor,
            pool_factors,
            self.data_idx,
            min_points=self.max_corr_min_points,
        )
        cand_factors = pool_factors + [factor]
        fit = self.trainer.gfit(
            cand_factors,
            self.returns,
            self.data_idx,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
        )
        score = fit.score
        if score is None:
            return float(self.compliance_fail_penalty)

        r["_score"] = float(score)
        r["_max_corr"] = max_c

        perf_reward = (1.0 - max_c) * float(score)
        return float(perf_reward * compliance)

    def run_batch(
        self,
        parallel_results: List[Dict[str, Any]],
        eval_indices: List[int],
        rewards: torch.Tensor,
        *,
        step: int = 0,
        run_dir: Optional[str] = None,
        formula_to_str: Optional[Callable[[List[int]], str]] = None,
    ) -> BatchSummary:
        """
        parallel_results 与 eval_indices 同序；在整批 reward 计算前固定当前 pool 快照（本 batch 内不更新 pool）。
        所有通过合规的候选（有 factor）均参与 pool 联合更新。
        返回 BatchSummary 供 engine 做日志输出。
        """
        pool_size_before = len(self.pool.entries)
        metrics_before = self._cached_pool_metrics
        if metrics_before is None and pool_size_before > 0:
            metrics_before = self._pool_metrics()

        pool_factors = self.pool.factor_series_list()
        candidates: List[Tuple[List[int], pd.Series]] = []

        for local_i, r in enumerate(parallel_results):
            idx = eval_indices[local_i]
            rw = self._reward_single(r, pool_factors)
            rewards[idx] = rw
            if r.get("ok") and "factor" in r and not r.get("missing_high") and not r.get("low_std"):
                candidates.append((r["formula"], r["factor"]))

        self._joint_update_pool(
            candidates,
            step=step,
            run_dir=run_dir,
            formula_to_str=formula_to_str,
        )

        pool_size_after = len(self.pool.entries)
        if candidates:
            metrics_after = self._pool_metrics()
        else:
            metrics_after = metrics_before
        self._cached_pool_metrics = metrics_after

        return BatchSummary(
            pool_size_before=pool_size_before,
            pool_size_after=pool_size_after,
            pool_before=metrics_before,
            pool_after=metrics_after,
            n_candidates=len(candidates),
        )

    def _joint_update_pool(
        self,
        candidates: List[Tuple[List[int], pd.Series]],
        *,
        step: int = 0,
        run_dir: Optional[str] = None,
        formula_to_str: Optional[Callable[[List[int]], str]] = None,
    ) -> None:
        if not candidates:
            return
        pool_entries = [(e.formula, e.factor) for e in self.pool.entries]
        all_entries = pool_entries + candidates
        factors = [fx for _, fx in all_entries]
        fit = self.trainer.gfit(
            factors,
            self.returns,
            self.data_idx,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
        )
        w = fit.weights
        if w.size == 0:
            return
        order = np.argsort(-np.abs(w))[: self.pool.capacity]
        kept = set(int(i) for i in order.tolist())
        removed_tuples = [all_entries[i] for i in range(len(all_entries)) if i not in kept]
        new_entries = [PoolEntry(formula=all_entries[i][0], factor=all_entries[i][1]) for i in order]
        self.pool.replace_entries(new_entries)
        if run_dir and formula_to_str and removed_tuples:
            removed_entries = [PoolEntry(formula=t[0], factor=t[1]) for t in removed_tuples]
            save_removed_pool_features(run_dir, step, removed_entries, formula_to_str, self.returns)
