import pandas as pd
import torch

from config import ModelConfig
from helpers.reward_metrics import (
    eval_final_reward_from_ics,
    evaluate_factor_full,
)


class AlphaBacktest:
    """
    评估规则改为基于 IC 绝对值：
    - 执行失败/约束失败：reward = penalty（默认 -5）
    - 通过后 reward = daily_icir*0.3 + monthly_icir*0.3 + overall_ic*0.4
    - use_smooth_reward=True 时：约束失败不再直接 -5，而是按合规程度线性插值
    评估时对齐 factor 与 returns 的 index。
    """

    def __init__(self, penalty=None, use_smooth_reward=True):
        if penalty is None:
            penalty = ModelConfig.BACKTEST_PENALTY
        self.penalty = penalty
        self.use_smooth_reward = use_smooth_reward
        self.icir_missing_gamma = float(ModelConfig.ICIR_MISSING_GAMMA)
        self.icir_missing_eps = float(ModelConfig.ICIR_MISSING_EPS)

    def eval_final_reward(
        self,
        daily_ic: pd.Series,
        monthly_ic: pd.Series,
        overall_ic: float,
    ):
        """返回 (score, daily_icir, monthly_icir)，与 helpers.reward_metrics 一致。"""
        return eval_final_reward_from_ics(
            daily_ic,
            monthly_ic,
            overall_ic,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
        )

    def evaluate(self, factor: pd.Series, returns: pd.Series):
        """
        factor: pd.Series 因子值，带时间索引
        returns: pd.Series 未来收益，带时间索引
        评估时按 index 对齐，取交集后计算 IC。
        """
        return evaluate_factor_full(
            factor,
            returns,
            penalty=self.penalty,
            use_smooth_reward=self.use_smooth_reward,
            icir_missing_gamma=self.icir_missing_gamma,
            icir_missing_eps=self.icir_missing_eps,
            device=ModelConfig.DEVICE,
        )
