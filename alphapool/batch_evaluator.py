"""
单轮 batch：并行得到 factor 后，按 alphapool 规则写 reward，并对有增益的因子联合训练后裁剪 pool。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from alphapool.correlation import max_abs_pearson
from alphapool.ensemble_trainer import AbstractEnsembleTrainer, LinearMeanStdEnsembleTrainer
from alphapool.pool_state import AlphaPoolState, PoolEntry
from config import ModelConfig


class AlphaPoolBatchEvaluator:
    def __init__(
        self,
        returns: pd.Series,
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
        pool: AlphaPoolState,
        trainer: AbstractEnsembleTrainer,
        *,
        icir_missing_gamma: float,
        icir_missing_eps: float,
        missing_threshold: float,
        unfinished_penalty: float,
        gain_eps: float = 1e-8,
        fragment_eval: bool = True,
    ):
        self.returns = returns
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.pool = pool
        self.trainer = trainer
        self.icir_missing_gamma = icir_missing_gamma
        self.icir_missing_eps = icir_missing_eps
        self.missing_threshold = missing_threshold
        self.unfinished_penalty = unfinished_penalty
        self.gain_eps = gain_eps
        self.fragment_eval = fragment_eval

    @classmethod
    def default_trainer(cls) -> LinearMeanStdEnsembleTrainer:
        return LinearMeanStdEnsembleTrainer(
            maxiter=int(getattr(ModelConfig, "ENSEMBLE_MAXITER", 400)),
        )

    def _baseline_test_score(self, pool_factors: List[pd.Series]) -> float:
        if not pool_factors:
            return 0.0
        fit = self.trainer.gfit(
            pool_factors,
            self.returns,
            self.train_idx,
            self.test_idx,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
            fragment_eval=self.fragment_eval,
        )
        return float(fit.test_score) if fit.test_score is not None else 0.0

    def _reward_single(self, r: Dict[str, Any], pool_factors: List[pd.Series]) -> float:
        if not r.get("ok", False):
            return float(self.unfinished_penalty)
        if r.get("missing_high"):
            return float(self.unfinished_penalty)
        if r.get("low_std"):
            return float(r.get("reward", self.unfinished_penalty))

        factor: pd.Series = r["factor"]
        compliance = float(r.get("compliance", 0.0))
        max_c = max_abs_pearson(factor, pool_factors, self.train_idx)
        cand_factors = pool_factors + [factor]
        fit = self.trainer.gfit(
            cand_factors,
            self.returns,
            self.train_idx,
            self.test_idx,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
            fragment_eval=self.fragment_eval,
        )
        test_score = fit.test_score
        if test_score is None:
            return float(self.unfinished_penalty)

        baseline = self._baseline_test_score(pool_factors)
        r["_gain"] = float(test_score) > baseline + self.gain_eps
        r["_test_score"] = float(test_score)
        r["_max_corr"] = max_c

        perf_reward = (1.0 - max_c) * float(test_score)
        return float(perf_reward * compliance)

    def run_batch(
        self,
        parallel_results: List[Dict[str, Any]],
        eval_indices: List[int],
        rewards: torch.Tensor,
    ) -> None:
        """
        parallel_results 与 eval_indices 同序；在整批 reward 计算前固定当前 pool 快照（本 batch 内不更新 pool）。
        """
        pool_factors = self.pool.factor_series_list()
        gainers: List[Tuple[List[int], pd.Series]] = []

        for local_i, r in enumerate(parallel_results):
            idx = eval_indices[local_i]
            rw = self._reward_single(r, pool_factors)
            rewards[idx] = rw
            if r.get("_gain"):
                gainers.append((r["formula"], r["factor"]))

        self._joint_update_pool(gainers)

    def _joint_update_pool(self, gainers: List[Tuple[List[int], pd.Series]]) -> None:
        if not gainers:
            return
        pool_entries = [(e.formula, e.factor) for e in self.pool.entries]
        all_entries = pool_entries + gainers
        factors = [fx for _, fx in all_entries]
        fit = self.trainer.gfit(
            factors,
            self.returns,
            self.train_idx,
            self.test_idx,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
            fragment_eval=self.fragment_eval,
        )
        w = fit.weights
        if w.size == 0:
            return
        order = np.argsort(-np.abs(w))[: self.pool.capacity]
        new_entries = [PoolEntry(formula=all_entries[i][0], factor=all_entries[i][1]) for i in order]
        self.pool.replace_entries(new_entries)
