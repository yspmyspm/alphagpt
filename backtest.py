import numpy as np
import pandas as pd
import torch

from config import ModelConfig
from helpers.backtest_helper import (
    check_finite_count,
    check_distribution,
    check_halflife,
    score_finite_ratio,
    score_distribution,
    score_halflife,
    calc_monthlyic,
    calc_dailyic,
    finite_rcor,
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
            penalty = getattr(ModelConfig, 'BACKTEST_PENALTY', -5.0)
        self.penalty = penalty
        self.use_smooth_reward = use_smooth_reward
        self.icir_missing_gamma = float(getattr(ModelConfig, 'ICIR_MISSING_GAMMA', 2.0))
        self.icir_missing_eps = float(getattr(ModelConfig, 'ICIR_MISSING_EPS', 1e-6))

    def _missing_penalty_factor(self, coverage: float) -> float:
        """coverage 越低，惩罚越强；gamma=2 时为二次惩罚。"""
        c = float(np.clip(coverage, 0.0, 1.0))
        return max(self.icir_missing_eps, c) ** self.icir_missing_gamma

    def eval_final_reward(
        self, 
        daily_ic: pd.Series,
        monthly_ic: pd.Series,
        overall_ic: float,
    ) -> float:
        daily_std = daily_ic.std()
        monthly_std = monthly_ic.std()
        daily_icir = daily_ic.mean() / daily_std if daily_std > 1e-8 else 0.0
        monthly_icir = monthly_ic.mean() / monthly_std if monthly_std > 1e-8 else 0.0

        daily_coverage = float(np.isfinite(daily_ic.to_numpy(dtype=np.float64)).mean()) if len(daily_ic) > 0 else 0.0
        monthly_coverage = float(np.isfinite(monthly_ic.to_numpy(dtype=np.float64)).mean()) if len(monthly_ic) > 0 else 0.0

        daily_icir *= self._missing_penalty_factor(daily_coverage)
        monthly_icir *= self._missing_penalty_factor(monthly_coverage)
        ic_score = overall_ic / (daily_std / 2 + monthly_std / 2 + 1e-8)

        score = 0
        score += abs(daily_icir) * 0.3
        score += abs(monthly_icir) * 0.3
        score += abs(ic_score) * 0.4
        return score, daily_icir, monthly_icir

    def evaluate(self, factor: pd.Series, returns: pd.Series):
        """
        factor: pd.Series 因子值，带时间索引
        returns: pd.Series 未来收益，带时间索引
        评估时按 index 对齐，取交集后计算 IC。
        """
        common = factor.index.intersection(returns.index)
        bad_reward_tuple = (
            torch.tensor(self.penalty, dtype=torch.float32, device=ModelConfig.DEVICE), 
            self.penalty, 
            float(self.penalty), 
            0.0, # daily_icir 
            0.0, # monthly_icir 
            0.0, # overall_ic
            0.0, # s_finite 
            0.0, # s_dist 
            0.0, # s_halflife 
            0.0, # compliance 
        )
        
        # print("common size", len(common))

        if len(common) == 0:
            return bad_reward_tuple

        common_len = len(common)
        f = factor.reindex(common).values.astype(np.float64)
        r = returns.reindex(common).values.astype(np.float64)
        mask = np.isfinite(f) & np.isfinite(r)
        if mask.sum() == 0:
            return bad_reward_tuple
        
        # print("available positions", mask.sum())

        f = f[mask]
        r = r[mask]
        common = common[mask]

        # 计算各约束的连续分数 [0, 1]
        s_finite = score_finite_ratio(f)
        s_dist = score_distribution(f)
        s_halflife = score_halflife(f)
        # print(s_finite, s_dist, s_halflife)


        if self.use_smooth_reward:
            compliance = s_finite * s_dist * s_halflife  # 乘积：任一接近 0 则 compliance 低
        else:
            if not check_finite_count(f) or not check_distribution(f) or not check_halflife(f):
                return bad_reward_tuple
            compliance = 1.0

        overall_ic = finite_rcor(f, r)
        if not np.isfinite(overall_ic):
            if self.use_smooth_reward:
                compliance *= 0.0
            else:
                return bad_reward_tuple

        monthly_ic = calc_monthlyic(pd.DataFrame({'factor': f, 'returns': r}, index=common))
        daily_ic = calc_dailyic(pd.DataFrame({'factor': f, 'returns': r}, index=common))

        good_reward, daily_icir, monthly_icir = self.eval_final_reward(daily_ic, monthly_ic, overall_ic)
        if not np.isfinite(good_reward):
            good_reward = 0.0

        if self.use_smooth_reward:
            # 线性插值：compliance=1 得 good_reward，compliance=0 得 penalty
            final_reward = self.penalty * (1.0 - compliance) + good_reward * compliance
        else:
            final_reward = good_reward

        return (
            torch.tensor(final_reward, dtype=torch.float32, device=ModelConfig.DEVICE),
            float(final_reward),
            daily_icir,
            monthly_icir,
            overall_ic,
            s_finite,
            s_dist,
            s_halflife,
            compliance,
        )
