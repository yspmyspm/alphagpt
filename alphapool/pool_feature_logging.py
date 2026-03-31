"""
Pool 内因子：公式（可读字符串）、ensemble 权重、单因子统计（IC / 分布 / 缺失 / autocorr）落盘。
当前 pool 与剔除项分目录存放。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from alphapool.pool_state import PoolEntry
from helpers.reward_metrics import compute_ic_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _formula_suffix(formula: str, used: Dict[str, int], *, max_len: int = 120) -> str:
    """将公式文本转为稳定文件后缀，避免 001 序号命名。"""
    normalized = " ".join(str(formula).strip().split())
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", normalized).strip("_")
    if not stem:
        stem = "formula"
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("_")
    base = f"{stem}__{digest}"
    n = used.get(base, 0) + 1
    used[base] = n
    if n == 1:
        return base
    return f"{base}__dup{n}"


def _plot_factor_distribution(fv: np.ndarray, path_raw: str, path_clipped: str, title: str) -> None:
    """每个因子两张分布图：原始 + clip；样本过少或分位数退化时仍写出 clipped 一份（与 raw 同数据或注明）。"""

    def _hist(data: np.ndarray, path: str, cap: str) -> None:
        n = len(data)
        fig, ax = plt.subplots(figsize=(8, 4))
        w = np.full(n, 1.0 / n, dtype=np.float64)
        dmin, dmax = float(np.min(data)), float(np.max(data))
        if dmax <= dmin:
            dmin -= 1.0
            dmax += 1.0
        ax.hist(data, bins=1000, range=(dmin, dmax), weights=w, density=False, alpha=0.7, edgecolor="none")
        ax.set_title(cap, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("count / n")
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
    """单因子的统计摘要：IC/ICIR/分布/自相关。"""
    vals = factor.values.astype(np.float64)
    finite_mask = np.isfinite(vals)
    n_finite = int(finite_mask.sum())
    stats: Dict[str, Any] = {}

    if n_finite > 1:
        fv = vals[finite_mask]
        s = pd.Series(fv)
        stats["mean"] = float(np.mean(fv))
        stats["std"] = float(np.std(fv, ddof=1))
        stats["skew"] = float(s.skew())
        stats["kurtosis"] = float(s.kurtosis())
        stats["quantile_info"] = [
            float(np.min(fv)),
            float(np.percentile(fv, 25)),
            float(np.median(fv)),
            float(np.percentile(fv, 75)),
            float(np.max(fv)),
        ]
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

    return stats


def _write_manifest(path: str, manifest: List[Dict[str, Any]]) -> None:
    """写 manifest，并将 quantile_info 压缩为单行列表。"""
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    lines = text.splitlines()
    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if '"quantile_info": [' not in line:
            out_lines.append(line)
            i += 1
            continue
        indent = line.split('"quantile_info"')[0]
        values: List[str] = []
        i += 1
        trailing = ""
        while i < len(lines):
            cur = lines[i].strip()
            if cur.startswith("]"):
                trailing = "," if cur.endswith(",") else ""
                break
            values.append(cur.rstrip(","))
            i += 1
        out_lines.append(f'{indent}"quantile_info": [{", ".join(values)}]{trailing}')
        i += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def save_pool_feature_snapshot(
    run_dir: str,
    step: int,
    entries: List[PoolEntry],
    formula_to_str: Callable[[List[int]], str],
    returns: pd.Series,
    *,
    subdir: str = "pool_features",
    weights: Optional[np.ndarray] = None,
    use_step_subdir: bool = True,
    include_feature_values: bool = True,
) -> Optional[str]:
    """
    将当前 pool 条目写入 run_dir/subdir[/step_XXXXXX]/：
    manifest.json + 分布图；可选保存因子数值 CSV。
    """
    if not entries:
        return None
    base = os.path.join(run_dir, subdir)
    out = os.path.join(base, f"step_{step:06d}") if use_step_subdir else base
    os.makedirs(out, exist_ok=True)
    values_dir = os.path.join(out, "feature_values")
    dist_dir = os.path.join(out, "feature_distributions")
    if not use_step_subdir:
        if os.path.isdir(dist_dir):
            shutil.rmtree(dist_dir)
        if os.path.isdir(values_dir):
            shutil.rmtree(values_dir)
    if include_feature_values:
        os.makedirs(values_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != len(entries):
            w = None

    manifest: List[Dict[str, Any]] = []
    used_suffixes: Dict[str, int] = {}
    for i, e in enumerate(entries):
        fname = formula_to_str(e.formula)
        suffix = _formula_suffix(fname, used_suffixes)
        row: Dict[str, Any] = {
            "index": i,
            "formula": fname,
            "file_suffix": suffix,
        }
        if w is not None:
            row["ensemble_weight"] = float(w[i])
        row["stats"] = _factor_stats(e.factor, returns)
        manifest.append(row)
        if include_feature_values:
            factor_rel = os.path.join("feature_values", f"factor_{suffix}.csv")
            pd.DataFrame({"factor": e.factor}).to_csv(os.path.join(out, factor_rel))
            row["factor_file"] = factor_rel

        fv = e.factor.values.astype(np.float64)
        fv = fv[np.isfinite(fv)]
        raw_rel = os.path.join("feature_distributions", f"dist_raw_{suffix}.png")
        clipped_rel = os.path.join("feature_distributions", f"dist_clipped_{suffix}.png")
        _plot_factor_distribution(
            fv,
            os.path.join(out, raw_rel),
            os.path.join(out, clipped_rel),
            fname,
        )
        row["distribution_raw_file"] = raw_rel
        row["distribution_clipped_file"] = clipped_rel

    _write_manifest(os.path.join(out, "manifest.json"), manifest)
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
    values_dir = os.path.join(out, "feature_values")
    dist_dir = os.path.join(out, "feature_distributions")
    os.makedirs(values_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    used_suffixes: Dict[str, int] = {}
    for i, e in enumerate(removed):
        fname = formula_to_str(e.formula)
        suffix = _formula_suffix(fname, used_suffixes)
        row: Dict[str, Any] = {
            "index": i,
            "formula": fname,
            "file_suffix": suffix,
            "stats": _factor_stats(e.factor, returns),
        }
        manifest.append(row)
        factor_rel = os.path.join("feature_values", f"factor_{suffix}.csv")
        pd.DataFrame({"factor": e.factor}).to_csv(os.path.join(out, factor_rel))
        row["factor_file"] = factor_rel

        fv = e.factor.values.astype(np.float64)
        fv = fv[np.isfinite(fv)]
        raw_rel = os.path.join("feature_distributions", f"dist_raw_{suffix}.png")
        clipped_rel = os.path.join("feature_distributions", f"dist_clipped_{suffix}.png")
        _plot_factor_distribution(
            fv,
            os.path.join(out, raw_rel),
            os.path.join(out, clipped_rel),
            fname,
        )
        row["distribution_raw_file"] = raw_rel
        row["distribution_clipped_file"] = clipped_rel

    _write_manifest(os.path.join(out, "manifest.json"), manifest)
    return out
