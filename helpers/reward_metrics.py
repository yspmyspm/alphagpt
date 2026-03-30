"""
可复用的因子评估指标：与 returns 对齐、合规分数、IC 序列、eval_final_reward 打分。
供 backtest.AlphaBacktest 与 alphapool 组合训练共用。
"""
from __future__ import annotations

from typing import Optional, Tuple

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


def missing_ratio_on_common(factor: pd.Series, returns: pd.Series) -> float:
    """在 factor 与 returns 索引交集上，factor 非有限值占比。"""
    common = factor.index.intersection(returns.index)
    if len(common) == 0:
        return 1.0
    f = factor.reindex(common).values.astype(np.float64)
    return float(np.mean(~np.isfinite(f)))


def align_factor_returns(
    factor: pd.Series, returns: pd.Series
) -> Optional[Tuple[pd.Index, np.ndarray, np.ndarray]]:
    """
    对齐 factor 与 returns，并应用双有限 mask。
    返回 (common_index, f, r) 或 None（无有效点）。
    """
    common = factor.index.intersection(returns.index)
    if len(common) == 0:
        return None
    f = factor.reindex(common).values.astype(np.float64)
    r = returns.reindex(common).values.astype(np.float64)
    mask = np.isfinite(f) & np.isfinite(r)
    if mask.sum() == 0:
        return None
    common = common[mask]

    return common, f[mask], r[mask]


def missing_penalty_factor(coverage: float, icir_missing_gamma: float, icir_missing_eps: float) -> float:
    c = float(np.clip(coverage, 0.0, 1.0))
    return max(icir_missing_eps, c) ** icir_missing_gamma


def eval_final_reward_from_ics(
    daily_ic: pd.Series,
    monthly_ic: pd.Series,
    overall_ic: float,
    *,
    icir_missing_gamma: float,
    icir_missing_eps: float,
) -> Tuple[float, float, float]:
    """
    与 AlphaBacktest.eval_final_reward 相同逻辑；返回 (score, daily_icir, monthly_icir)。
    score 为加权 IC 指标，可用于 alphapool 中的 performance。
    """
    daily_std = daily_ic.std()
    monthly_std = monthly_ic.std()
    std_eps = float(ModelConfig.ICIR_STD_EPS)
    daily_icir = daily_ic.mean() / daily_std if daily_std > std_eps else 0.0
    monthly_icir = monthly_ic.mean() / monthly_std if monthly_std > std_eps else 0.0

    daily_coverage = float(np.isfinite(daily_ic.to_numpy(dtype=np.float64)).mean()) if len(daily_ic) > 0 else 0.0
    monthly_coverage = float(np.isfinite(monthly_ic.to_numpy(dtype=np.float64)).mean()) if len(monthly_ic) > 0 else 0.0

    daily_icir *= missing_penalty_factor(daily_coverage, icir_missing_gamma, icir_missing_eps)
    monthly_icir *= missing_penalty_factor(monthly_coverage, icir_missing_gamma, icir_missing_eps)
    # 分母为日度/月度 IC 序列标准差的几何平均（替代算术平均）
    geom_eps = float(ModelConfig.GEOM_STD_EPS)
    ic_div_eps = float(ModelConfig.IC_SCORE_DIV_EPS)
    geom_std = float(np.sqrt(max(daily_std * monthly_std, 0.0) + geom_eps))
    ic_score = overall_ic / (geom_std + ic_div_eps)

    w_daily = float(ModelConfig.REWARD_WEIGHT_DAILY_ICIR)
    w_monthly = float(ModelConfig.REWARD_WEIGHT_MONTHLY_ICIR)
    w_ic = float(ModelConfig.REWARD_WEIGHT_IC_SCORE)
    score = abs(daily_icir) * w_daily + abs(monthly_icir) * w_monthly + abs(ic_score) * w_ic
    return score, daily_icir, monthly_icir


def compute_ic_metrics(
    factor: pd.Series,
    returns: pd.Series,
    *,
    subset_index: Optional[pd.Index] = None,
) -> Optional[Tuple[pd.Series, pd.Series, float]]:
    """
    计算 daily_ic / monthly_ic / overall_ic。
    若给定 subset_index，只在该子集上评估（用于 train/test 切片）。
    """
    if subset_index is not None:
        factor = factor.reindex(subset_index)
        returns = returns.reindex(subset_index)
    aligned = align_factor_returns(factor, returns)
    if aligned is None:
        return None
    common, f, r = aligned
    overall_ic = finite_rcor(f, r)
    if not np.isfinite(overall_ic):
        overall_ic = float("nan")
    df = pd.DataFrame({"factor": f, "returns": r}, index=common)
    monthly_ic = calc_monthlyic(df)
    daily_ic = calc_dailyic(df)
    return daily_ic, monthly_ic, float(overall_ic)


def compute_compliance_only(
    factor: pd.Series,
    returns: pd.Series,
    *,
    use_smooth_reward: bool,
) -> Optional[Tuple[float, float, float, float]]:
    """
    仅计算合规相关分数；若无法对齐或无有效 IC，返回 None。
    返回 (s_finite, s_dist, s_halflife, compliance)。
    """
    aligned = align_factor_returns(factor, returns)


    if aligned is None:
        return None
    common, f, r = aligned


    s_finite = score_finite_ratio(f)
    s_dist = score_distribution(f)
    s_halflife = score_halflife(f)

    if use_smooth_reward:
        compliance = s_finite * s_dist * s_halflife
    else:
        if not check_finite_count(f) or not check_distribution(f) or not check_halflife(f):
            return None
        compliance = 1.0

    overall_ic = finite_rcor(f, r)
    if not np.isfinite(overall_ic):
        if use_smooth_reward:
            compliance *= 0.0
        else:
            return None


    return float(s_finite), float(s_dist), float(s_halflife), float(compliance)


def performance_score_on_subset(
    factor: pd.Series,
    returns: pd.Series,
    subset_index: pd.Index,
    *,
    icir_missing_gamma: float,
    icir_missing_eps: float,
    eval_mode: str = "fragment_ic",
) -> Optional[float]:
    """
    eval_mode:
      - fragment_ic: 短片段 / 单月 / CV 折 —— 仅用 |overall_ic| 作为打分（训练目标与测试片段一致）。
      - full_weighted: 与 eval_final_reward_from_ics 一致（用于完整样本上的组合评估）。
    """
    ic = compute_ic_metrics(factor, returns, subset_index=subset_index)
    if ic is None:
        return None
    daily_ic, monthly_ic, overall_ic = ic
    if not np.isfinite(overall_ic):
        return None
    if eval_mode == "fragment_ic":
        return float(abs(overall_ic))
    if eval_mode != "full_weighted":
        raise ValueError(f"Invalid eval_mode: {eval_mode!r}, expected 'fragment_ic' or 'full_weighted'")
    score, _, _ = eval_final_reward_from_ics(
        daily_ic, monthly_ic, overall_ic,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
    )
    return float(score) if np.isfinite(score) else None


def metrics_from_factor_returns_full(
    factor: pd.Series,
    returns: pd.Series,
    *,
    icir_missing_gamma: float,
    icir_missing_eps: float,
) -> Optional[dict]:
    """
    完整对齐样本上的 IC 指标（不含 compliance 加权前的 good_reward）。
    用于曲线：overall_ic, daily_icir, monthly_icir, weighted_score（eval_final_reward 主项）。
    """
    ic = compute_ic_metrics(factor, returns, subset_index=None)
    if ic is None:
        return None
    daily_ic, monthly_ic, overall_ic = ic
    if not np.isfinite(overall_ic):
        return None
    weighted_score, daily_icir, monthly_icir = eval_final_reward_from_ics(
        daily_ic, monthly_ic, overall_ic,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
    )
    return {
        "overall_ic": float(overall_ic),
        "daily_icir": float(daily_icir),
        "monthly_icir": float(monthly_icir),
        "weighted_score": float(weighted_score),
    }


def evaluate_factor_full(
    factor: pd.Series,
    returns: pd.Series,
    *,
    penalty: float,
    use_smooth_reward: bool,
    icir_missing_gamma: float,
    icir_missing_eps: float,
    device: Optional[torch.device] = None,
):
    """
    与 AlphaBacktest.evaluate 等价：返回
    (torch_reward, float_reward, daily_icir, monthly_icir, overall_ic, s_fin, s_dist, s_halflife, compliance)
    """
    if device is None:
        device = ModelConfig.DEVICE
    bad = (
        torch.tensor(penalty, dtype=torch.float32, device=device),
        penalty,
        float(penalty),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    aligned = align_factor_returns(factor, returns)
    if aligned is None:
        return bad

    common, f, r = aligned
    s_finite = score_finite_ratio(f)
    s_dist = score_distribution(f)
    s_halflife = score_halflife(f)

    if use_smooth_reward:
        compliance = s_finite * s_dist * s_halflife
    else:
        if not check_finite_count(f) or not check_distribution(f) or not check_halflife(f):
            return bad
        compliance = 1.0

    overall_ic = finite_rcor(f, r)
    if not np.isfinite(overall_ic):
        if use_smooth_reward:
            compliance *= 0.0
        else:
            return bad

    monthly_ic = calc_monthlyic(pd.DataFrame({"factor": f, "returns": r}, index=common))
    daily_ic = calc_dailyic(pd.DataFrame({"factor": f, "returns": r}, index=common))

    good_reward, daily_icir, monthly_icir = eval_final_reward_from_ics(
        daily_ic, monthly_ic, overall_ic,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
    )
    if not np.isfinite(good_reward):
        good_reward = 0.0

    if use_smooth_reward:
        final_reward = penalty * (1.0 - compliance) + good_reward * compliance
    else:
        final_reward = good_reward

    return (
        torch.tensor(final_reward, dtype=torch.float32, device=device),
        float(final_reward),
        float(daily_icir),
        float(monthly_icir),
        float(overall_ic),
        float(s_finite),
        float(s_dist),
        float(s_halflife),
        float(compliance),
    )
