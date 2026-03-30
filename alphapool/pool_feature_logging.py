"""
Pool 内因子：公式（可读字符串）、ensemble 权重、单因子统计（IC / 分布 / 缺失 / autocorr）落盘。
当前 pool 与剔除项分目录存放。
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from alphapool.pool_state import PoolEntry
from helpers.reward_metrics import compute_ic_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _plot_factor_distribution(fv: np.ndarray, path_raw: str, path_clipped: str, title: str) -> None:
    """每个因子画两张分布图：原始 + clip 到 q01-q99。"""
    if len(fv) < 10:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fv, bins=1000, density=True, alpha=0.7, edgecolor="none")
    ax.set_title(f"Distribution (raw): {title}", fontsize=9)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_raw, dpi=100)
    plt.close(fig)

    q01, q99 = np.percentile(fv, 1), np.percentile(fv, 99)
    if q01 >= q99:
        return
    fv_c = np.clip(fv, q01, q99)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fv_c, bins=1000, density=True, alpha=0.7, edgecolor="none")
    ax.set_title(f"Distribution (q01\u2013q99): {title}", fontsize=9)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_clipped, dpi=100)
    plt.close(fig)


def _factor_stats(factor: pd.Series, returns: pd.Series) -> Dict[str, Any]:
    """单因子的统计摘要：IC/ICIR/分布/缺失率/自相关。"""
    vals = factor.values.astype(np.float64)
    finite_mask = np.isfinite(vals)
    n_total = len(vals)
    n_finite = int(finite_mask.sum())
    missing_ratio = 1.0 - n_finite / n_total if n_total > 0 else 1.0

    stats: Dict[str, Any] = {
        "n_total": n_total,
        "n_finite": n_finite,
        "missing_ratio": round(missing_ratio, 6),
    }

    if n_finite > 1:
        fv = vals[finite_mask]
        s = pd.Series(fv)
        stats["mean"] = float(np.mean(fv))
        stats["std"] = float(np.std(fv, ddof=1))
        stats["skew"] = float(s.skew())
        stats["kurtosis"] = float(s.kurtosis())
        stats["min"] = float(np.min(fv))
        stats["q25"] = float(np.percentile(fv, 25))
        stats["median"] = float(np.median(fv))
        stats["q75"] = float(np.percentile(fv, 75))
        stats["max"] = float(np.max(fv))
        ac = factor.autocorr(lag=5)
        stats["autocorr_5"] = float(ac) if np.isfinite(ac) else None

    ic = compute_ic_metrics(factor, returns)
    if ic is not None:
        daily_ic, monthly_ic, overall_ic = ic
        d_std = daily_ic.std()
        m_std = monthly_ic.std()
        stats["overall_ic"] = float(overall_ic)
        stats["daily_icir"] = float(daily_ic.mean() / d_std) if d_std > 1e-8 else 0.0
        stats["monthly_icir"] = float(monthly_ic.mean() / m_std) if m_std > 1e-8 else 0.0
        stats["daily_ic_coverage"] = (
            float(np.isfinite(daily_ic.to_numpy(dtype=np.float64)).mean()) if len(daily_ic) > 0 else 0.0
        )
        stats["monthly_ic_coverage"] = (
            float(np.isfinite(monthly_ic.to_numpy(dtype=np.float64)).mean()) if len(monthly_ic) > 0 else 0.0
        )

    return stats


def save_pool_feature_snapshot(
    run_dir: str,
    step: int,
    entries: List[PoolEntry],
    formula_to_str: Callable[[List[int]], str],
    returns: pd.Series,
    *,
    subdir: str = "pool_features",
    weights: Optional[np.ndarray] = None,
) -> Optional[str]:
    """
    将当前 pool 条目写入 run_dir/subdir/step_XXXXXX/：
    manifest.json（公式、ensemble 权重、单因子统计）+ factor_XXX.csv。
    """
    if not entries:
        return None
    base = os.path.join(run_dir, subdir)
    out = os.path.join(base, f"step_{step:06d}")
    os.makedirs(out, exist_ok=True)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != len(entries):
            w = None

    manifest: List[Dict[str, Any]] = []
    for i, e in enumerate(entries):
        fname = formula_to_str(e.formula)
        row: Dict[str, Any] = {
            "index": i,
            "formula": fname,
        }
        if w is not None:
            row["ensemble_weight"] = float(w[i])
        row["stats"] = _factor_stats(e.factor, returns)
        manifest.append(row)
        pd.DataFrame({"factor": e.factor}).to_csv(os.path.join(out, f"factor_{i:03d}.csv"))

        fv = e.factor.values.astype(np.float64)
        fv = fv[np.isfinite(fv)]
        _plot_factor_distribution(
            fv,
            os.path.join(out, f"dist_raw_{i:03d}.png"),
            os.path.join(out, f"dist_clipped_{i:03d}.png"),
            fname,
        )

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out


def save_removed_pool_features(
    run_dir: str,
    step: int,
    removed: List[PoolEntry],
    formula_to_str: Callable[[List[int]], str],
    returns: pd.Series,
) -> Optional[str]:
    """被 joint 更新剔除的因子写入 run_dir/pool_features_removed/step_XXXXXX/。"""
    if not removed:
        return None
    base = os.path.join(run_dir, "pool_features_removed")
    out = os.path.join(base, f"step_{step:06d}")
    os.makedirs(out, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    for i, e in enumerate(removed):
        fname = formula_to_str(e.formula)
        row: Dict[str, Any] = {
            "index": i,
            "formula": fname,
            "stats": _factor_stats(e.factor, returns),
        }
        manifest.append(row)
        pd.DataFrame({"factor": e.factor}).to_csv(os.path.join(out, f"factor_{i:03d}.csv"))

        fv = e.factor.values.astype(np.float64)
        fv = fv[np.isfinite(fv)]
        _plot_factor_distribution(
            fv,
            os.path.join(out, f"dist_raw_{i:03d}.png"),
            os.path.join(out, f"dist_clipped_{i:03d}.png"),
            fname,
        )

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out
