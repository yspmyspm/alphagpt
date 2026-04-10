"""
AlphaPool 因子评估指标（IC、合规、对齐等）。

实现位于当前仓库的 ``AlphaPool/utils``，供 ``worker``、日志与引擎共用。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ── 基础工具 ─────────────────────────────────────────────

def r_cor(f1: np.ndarray, f2: np.ndarray) -> float:
    if np.dot(f1, f1) == 0 or np.dot(f2, f2) == 0:
        return 0.0
    return float(np.dot(f1, f2) / np.sqrt(np.dot(f1, f1) * np.dot(f2, f2)))


def finite_rcor(f1, f2) -> float:
    a = f1.values if isinstance(f1, pd.Series) else f1
    b = f2.values if isinstance(f2, pd.Series) else f2
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return 0.0
    return r_cor(a[mask], b[mask])


def align_factor_returns(
    factor: pd.Series, returns: pd.Series,
) -> Optional[Tuple[pd.Index, np.ndarray, np.ndarray]]:
    common = factor.index.intersection(returns.index)
    if len(common) == 0:
        return None
    f = factor.reindex(common).values.astype(np.float64)
    r = returns.reindex(common).values.astype(np.float64)
    mask = np.isfinite(f) & np.isfinite(r)
    if mask.sum() == 0:
        return None
    return common[mask], f[mask], r[mask]


def missing_ratio_on_common(factor: pd.Series, returns: pd.Series) -> float:
    common = factor.index.intersection(returns.index)
    if len(common) == 0:
        return 1.0
    f = factor.reindex(common).values.astype(np.float64)
    return float(np.mean(~np.isfinite(f)))


# ── IC 序列 ──────────────────────────────────────────────

def calc_monthlyic(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    values = df[["factor", "returns"]].to_numpy(dtype=float)
    f, r = values[:, 0], values[:, 1]
    month_codes = df.index.values.astype("datetime64[M]").astype("int64")
    unique_months, inv = np.unique(month_codes, return_inverse=True)
    sum_fr = np.bincount(inv, weights=f * r)
    sum_f2 = np.bincount(inv, weights=f * f)
    sum_r2 = np.bincount(inv, weights=r * r)
    denom = np.sqrt(sum_f2) * np.sqrt(sum_r2)
    monthly_ic_values = np.divide(
        sum_fr, denom,
        out=np.full_like(sum_fr, np.nan, dtype=float),
        where=(denom != 0) & np.isfinite(denom),
    )
    return pd.Series(
        monthly_ic_values,
        index=pd.to_datetime(unique_months, unit="M").strftime("%Y-%m"),
    )


def calc_dailyic(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    values = df[["factor", "returns"]].to_numpy(dtype=float)
    f, r = values[:, 0], values[:, 1]
    day_codes = df.index.values.astype("datetime64[D]").astype("int64")
    unique_days, inv = np.unique(day_codes, return_inverse=True)
    sum_fr = np.bincount(inv, weights=f * r)
    sum_f2 = np.bincount(inv, weights=f * f)
    sum_r2 = np.bincount(inv, weights=r * r)
    denom = np.sqrt(sum_f2) * np.sqrt(sum_r2)
    daily_ic_values = np.divide(
        sum_fr, denom,
        out=np.full_like(sum_fr, np.nan, dtype=float),
        where=(denom != 0) & np.isfinite(denom),
    )
    return pd.Series(daily_ic_values, index=pd.to_datetime(unique_days, unit="D").date)


# ── 合规分数 ─────────────────────────────────────────────

def check_finite_count(arr: np.ndarray, min_ratio: float = 0.8) -> bool:
    return float(np.isfinite(arr).sum() / max(1, arr.shape[0])) >= min_ratio


def score_finite_ratio(arr: np.ndarray, min_ratio: float = 0.8) -> float:
    ratio = np.isfinite(arr).sum() / max(1, arr.shape[0])
    return min(1.0, ratio / min_ratio)


def check_distribution(arr: np.ndarray, skew_limit: float = 10.0, kurt_limit: float = 100.0) -> bool:
    fv = (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-8)
    skew = pd.Series(fv).skew()
    kurt = pd.Series(fv).kurt()
    if np.isnan(skew) or np.isnan(kurt):
        return False
    if abs(skew) > skew_limit or abs(kurt) > kurt_limit:
        return False
    return np.nanstd(fv) != 0


def score_distribution(arr: np.ndarray, skew_limit: float = 20.0, kurt_limit: float = 200.0) -> float:
    fv = (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-8)
    skew = pd.Series(fv).skew()
    kurt = pd.Series(fv).kurt()
    if np.isnan(skew) or np.isnan(kurt) or np.nanstd(fv) == 0:
        return 0.0
    return min(max(0.0, 1.0 - abs(skew) / skew_limit), max(0.0, 1.0 - abs(kurt) / kurt_limit))


def check_halflife(arr: np.ndarray, min_corr: float = 0.1, lag: int = 1) -> bool:
    if min_corr <= -1.0:
        return True
    shifted = np.roll(arr, lag)
    shifted[:lag] = np.nan
    return finite_rcor(arr, shifted) >= min_corr


def score_halflife(arr: np.ndarray, min_corr: float = 0.1, lag: int = 1) -> float:
    if min_corr <= -1.0:
        return 1.0
    shifted = np.roll(arr, lag)
    shifted[:lag] = np.nan
    corr = finite_rcor(arr, shifted)
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return min(1.0, max(0.0, (corr + 1.0) / (min_corr + 1.0)))


# ── IC 指标聚合 ──────────────────────────────────────────

def missing_penalty_factor(coverage: float, gamma: float = 2.0, eps: float = 1e-6) -> float:
    c = float(np.clip(coverage, 0.0, 1.0))
    return max(eps, c) ** gamma


def compute_ic_metrics(
    factor: pd.Series,
    returns: pd.Series,
    *,
    subset_index: Optional[pd.Index] = None,
) -> Optional[Tuple[pd.Series, pd.Series, float]]:
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
    return calc_dailyic(df), calc_monthlyic(df), float(overall_ic)


def compute_ic_score_details(
    daily_ic: pd.Series,
    monthly_ic: pd.Series,
    overall_ic: float,
    *,
    icir_missing_gamma: float = 2.0,
    icir_missing_eps: float = 1e-6,
    icir_std_eps: float = 1e-8,
    geom_std_eps: float = 1e-16,
    ic_score_div_eps: float = 1e-8,
    w_daily: float = 0.3,
    w_monthly: float = 0.3,
    w_ic: float = 0.4,
) -> dict:
    daily_std = daily_ic.std()
    monthly_std = monthly_ic.std()
    if not np.isfinite(daily_std):
        daily_std = 0.0
    if not np.isfinite(monthly_std):
        monthly_std = 0.0
    daily_mean = float(daily_ic.mean()) if len(daily_ic) > 0 else 0.0
    monthly_mean = float(monthly_ic.mean()) if len(monthly_ic) > 0 else 0.0
    daily_icir_raw = daily_mean / daily_std if daily_std > icir_std_eps else 0.0
    monthly_icir_raw = monthly_mean / monthly_std if monthly_std > icir_std_eps else 0.0

    d_cov = float(np.isfinite(daily_ic.to_numpy(dtype=np.float64)).mean()) if len(daily_ic) > 0 else 0.0
    m_cov = float(np.isfinite(monthly_ic.to_numpy(dtype=np.float64)).mean()) if len(monthly_ic) > 0 else 0.0
    daily_missing_penalty = missing_penalty_factor(d_cov, icir_missing_gamma, icir_missing_eps)
    monthly_missing_penalty = missing_penalty_factor(m_cov, icir_missing_gamma, icir_missing_eps)
    daily_icir = daily_icir_raw * daily_missing_penalty
    monthly_icir = monthly_icir_raw * monthly_missing_penalty

    geom_product = daily_std * monthly_std
    if not np.isfinite(geom_product):
        geom_product = 0.0
    geom_std = float(np.sqrt(max(geom_product, 0.0) + geom_std_eps))
    ic_score = overall_ic / (geom_std + ic_score_div_eps)

    score = abs(daily_icir) * w_daily + abs(monthly_icir) * w_monthly + abs(ic_score) * w_ic
    return {
        "overall_ic": float(overall_ic),
        "daily_ic_mean": float(daily_mean),
        "monthly_ic_mean": float(monthly_mean),
        "daily_ic_std": float(daily_std),
        "monthly_ic_std": float(monthly_std),
        "daily_icir_raw": float(daily_icir_raw),
        "monthly_icir_raw": float(monthly_icir_raw),
        "daily_coverage": float(d_cov),
        "monthly_coverage": float(m_cov),
        "daily_missing_penalty": float(daily_missing_penalty),
        "monthly_missing_penalty": float(monthly_missing_penalty),
        "daily_icir": float(daily_icir),
        "monthly_icir": float(monthly_icir),
        "ic_score": float(ic_score),
        "combined_score": float(score),
    }


def compute_compliance_only(
    factor: pd.Series,
    returns: pd.Series,
    *,
    use_smooth_reward: bool = True,
    min_finite_ratio: float = 0.8,
    skew_limit_check: float = 10.0,
    kurt_limit_check: float = 100.0,
    skew_limit_score: float = 20.0,
    kurt_limit_score: float = 200.0,
    halflife_min_corr: float = 0.1,
    halflife_lag: int = 1,
) -> Optional[Tuple[float, float, float, float]]:
    """Returns (s_finite, s_dist, s_halflife, compliance) or None."""
    aligned = align_factor_returns(factor, returns)
    if aligned is None:
        return None
    common, f, r = aligned

    s_finite = score_finite_ratio(f, min_ratio=min_finite_ratio)
    s_dist = score_distribution(f, skew_limit=skew_limit_score, kurt_limit=kurt_limit_score)
    s_half = score_halflife(f, min_corr=halflife_min_corr, lag=halflife_lag)

    if use_smooth_reward:
        compliance = s_finite * s_dist * s_half
    else:
        if (
            not check_finite_count(f, min_ratio=min_finite_ratio)
            or not check_distribution(f, skew_limit=skew_limit_check, kurt_limit=kurt_limit_check)
            or not check_halflife(f, min_corr=halflife_min_corr, lag=halflife_lag)
        ):
            return None
        compliance = 1.0

    overall_ic = finite_rcor(f, r)
    if not np.isfinite(overall_ic):
        if use_smooth_reward:
            compliance *= 0.0
        else:
            return None

    return float(s_finite), float(s_dist), float(s_half), float(compliance)
