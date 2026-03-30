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
    """每个因子两张分布图：原始 + clip；样本过少或分位数退化时仍写出 clipped 一份（与 raw 同数据或注明）。"""

    def _hist(data: np.ndarray, path: str, cap: str) -> None:
        nb = min(1000, max(20, len(data) // 10)) if len(data) > 0 else 10
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=nb, density=True, alpha=0.7, edgecolor="none")
        ax.set_title(cap, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    if len(fv) == 0:
        z = np.zeros(1)
        _hist(z, path_raw, f"Distribution (raw, empty): {title}")
        _hist(z, path_clipped, f"Distribution (clipped, empty): {title}")
        return

    if len(fv) < 10:
        _hist(fv, path_raw, f"Distribution (raw, n={len(fv)}): {title}")
        _hist(fv, path_clipped, f"Distribution (clipped=same as raw, n={len(fv)}): {title}")
        return

    _hist(fv, path_raw, f"Distribution (raw): {title}")
    q01, q99 = np.percentile(fv, 1), np.percentile(fv, 99)
    if q01 >= q99:
        _hist(fv, path_clipped, f"Distribution (clipped=same as raw, degenerate q01>=q99): {title}")
        return
    fv_c = np.clip(fv, q01, q99)
    _hist(fv_c, path_clipped, f"Distribution (q01\u2013q99): {title}")


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
