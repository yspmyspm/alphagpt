"""
每轮训练后：保存 pool 线性组合预测、在全样本上做完整 evaluate，
并在 pool_ic/ 下按指标分别绘制曲线（overall_ic、dICIR、mICIR、weighted_score，各 raw/abs）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from alphapool.ensemble_trainer import build_full_prediction, LinearMeanStdEnsembleTrainer
from alphapool.pool_feature_logging import save_pool_feature_snapshot
from alphapool.pool_state import AlphaPoolState
from backtest import AlphaBacktest
from helpers.reward_metrics import evaluate_factor_full, metrics_from_factor_returns_full


def _plot_pool_metric_series(rows: List[Dict[str, Any]], pool_ic_dir: str) -> None:
    """每个指标单独一张图：overall_ic、dICIR、mICIR、weighted_score；各含 raw 与 abs 版本。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    if not rows:
        return

    steps = [r["step"] for r in rows]
    series = [
        ("overall_ic", [float(r["overall_ic"]) for r in rows], "overall_ic"),
        ("daily_icir", [float(r["daily_icir"]) for r in rows], "dICIR (daily ICIR)"),
        ("monthly_icir", [float(r["monthly_icir"]) for r in rows], "mICIR (monthly ICIR)"),
        ("weighted_score", [float(r["weighted_score"]) for r in rows], "weighted_score"),
    ]
    for fname, ys, label in series:
        for use_abs, suf in ((False, "raw"), (True, "abs")):
            arr = np.asarray(ys, dtype=float)
            yplot = np.abs(arr) if use_abs else arr
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(steps, yplot, linewidth=1.2, color="C0")
            ax.set_xlabel("train_step")
            ax.set_ylabel("value")
            ax.set_title(f"{label} ({suf})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(pool_ic_dir, f"{fname}_{suf}.png"), dpi=120)
            plt.close(fig)


def update_pool_monitoring(
    step: int,
    run_dir: str,
    pool: AlphaPoolState,
    returns: pd.Series,
    data_idx: pd.Index,
    trainer: LinearMeanStdEnsembleTrainer,
    *,
    icir_missing_gamma: float,
    icir_missing_eps: float,
    bt: AlphaBacktest,
    history: List[Dict[str, Any]],
    formula_to_str: Optional[Callable[[List[int]], str]] = None,
) -> None:
    """
    用当前 pool 在全体数据上拟合权重，生成全时段线性预测；
    与 returns 对齐后，用完整 evaluate（含 compliance）与 IC 分解指标记录并落盘。
    """
    factors = pool.factor_series_list()
    pool_dir = os.path.join(run_dir, "pool_artifacts")
    pool_ic_dir = os.path.join(run_dir, "pool_ic")
    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(pool_ic_dir, exist_ok=True)

    if not factors or len(data_idx) == 0:
        return

    fit = trainer.gfit(
        factors,
        returns,
        data_idx,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
    )
    if formula_to_str is not None and pool.entries:
        save_pool_feature_snapshot(
            run_dir,
            step,
            pool.entries,
            formula_to_str,
            returns,
            weights=fit.weights,
        )
    pred = build_full_prediction(factors, returns, fit.weights)
    if len(pred) == 0:
        return

    ret_a = returns.reindex(pred.index)

    m = metrics_from_factor_returns_full(
        pred,
        ret_a,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
    )
    if m is None:
        return

    pred_step_path = os.path.join(pool_dir, f"prediction_step_{step:06d}.csv")
    pred_ret_step_path = os.path.join(pool_dir, f"prediction_returns_step_{step:06d}.csv")
    pd.DataFrame({"prediction": pred}).to_csv(pred_step_path)
    pd.DataFrame({"prediction": pred, "returns": ret_a}).to_csv(pred_ret_step_path)
    pd.DataFrame({"prediction": pred}).to_csv(os.path.join(pool_dir, "prediction_latest.csv"))
    pd.DataFrame({"prediction": pred, "returns": ret_a}).to_csv(
        os.path.join(pool_dir, "prediction_returns_latest.csv")
    )

    ev = evaluate_factor_full(
        pred,
        ret_a,
        penalty=bt.penalty,
        use_smooth_reward=bt.use_smooth_reward,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
    )
    _, final_reward, daily_icir, monthly_icir, overall_ic_ev, s_fin, s_dist, s_halflife, compliance = ev

    icir_mean = (abs(float(daily_icir)) + abs(float(monthly_icir))) / 2.0

    row: Dict[str, Any] = {
        "step": int(step),
        "overall_ic": float(m["overall_ic"]),
        "daily_icir": float(m["daily_icir"]),
        "monthly_icir": float(m["monthly_icir"]),
        "weighted_score": float(m["weighted_score"]),
        "icir_mean": float(icir_mean),
        "overall_ic_eval": float(overall_ic_ev),
        "final_reward": float(final_reward),
        "compliance": float(compliance),
        "s_finite": float(s_fin),
        "s_dist": float(s_dist),
        "s_halflife": float(s_halflife),
        "pool_size": len(factors),
    }
    history.append(row)

    with open(os.path.join(pool_ic_dir, "pool_metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    _plot_pool_metric_series(history, pool_ic_dir)
