"""因子值分布直方图（原始 + 分位裁剪），供 pool 快照落盘。"""
from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_factor_distribution(
    fv: np.ndarray,
    path_raw: str,
    path_clipped: str,
    title: str,
) -> None:
    """每个因子两张分布图：原始 + clip；样本过少或分位数退化时仍写出 clipped 一份。"""

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
    _hist(fv_c, path_clipped, f"Distribution (q01–q99): {title}")
