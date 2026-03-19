import numpy as np
import pandas as pd
import torch

from config import ModelConfig
from helpers.backtest_helper import *




class AlphaBacktest:
    """
    评估规则改为基于 IC 绝对值：
    - 执行失败/约束失败：reward = -5
    - 通过后 reward = clip(|IC|, 0, 0.2) * 20
    评估时对齐 factor 与 returns 的 index。
    """

    def __init__(self, penalty=-5.0):
        self.penalty = penalty

    def eval_final_reward(self, daily_icir: float, monthly_icir: float, overall_ic: float) -> float:
        return daily_icir * 0.3 + monthly_icir * 0.3 + overall_ic * 0.4

    def evaluate(self, factor: pd.Series, returns: pd.Series):
        """
        factor: pd.Series 因子值，带时间索引
        returns: pd.Series 未来收益，带时间索引
        评估时按 index 对齐，取交集后计算 IC。
        """
        common = factor.index.intersection(returns.index)
        bad_reward = (torch.tensor(self.penalty, dtype=torch.float32, device=ModelConfig.DEVICE), self.penalty, 0.0, 0.0, 0.0)
        
        if len(common) == 0:
            return bad_reward

        f = factor.reindex(common).values.astype(np.float64)
        r = returns.reindex(common).values.astype(np.float64)
        mask = np.isfinite(f) & np.isfinite(r)
        if mask.sum() == 0:
            return bad_reward
        
        f = f[mask]
        r = r[mask]
        common = common[mask]

        if not check_finite_count(f):
            return bad_reward

        if not check_distribution(f):
            return bad_reward
        

        if not check_halflife(f):
            return bad_reward


        overall_ic = finite_rcor(f, r)
        if not np.isfinite(overall_ic):
            return bad_reward

        monthly_ic = calc_monthlyic(pd.DataFrame({'factor': f, 'returns': r}, index = common))
        monthly_icir = monthly_ic.mean() / monthly_ic.std()
        

        daily_ic = calc_dailyic(pd.DataFrame({'factor': f, 'returns': r}, index = common))
        daily_icir = daily_ic.mean() / daily_ic.std()

        # 对 [daily_ic, monthly_ic, ic] 进行综合评价，给出最终奖励。
        
        final_reward = self.eval_final_reward(daily_icir, monthly_icir, overall_ic)
        return torch.tensor(final_reward, dtype=torch.float32, device=ModelConfig.DEVICE), float(final_reward), daily_icir, monthly_icir, overall_ic
