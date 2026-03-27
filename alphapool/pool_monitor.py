"""
每轮训练后：保存 pool 线性组合预测、在全样本上做完整 evaluate，
并更新 overall_ic / ICIR 等曲线图（raw 与绝对值）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from alphapool.ensemble_trainer import build_full_prediction, LinearMeanStdEnsembleTrainer
from alphapool.pool_state import AlphaPoolState
from backtest import AlphaBacktest
from helpers.reward_metrics import evaluate_factor_full, metrics_from_factor_returns_full


def _plot_ic_curves(rows: List[Dict[str, Any]], path_raw: str, path_abs: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    if not rows:
        return

    steps = [r["step"] for r in rows]
    oic = [r["overall_ic"] for r in rows]
    dic = [r["daily_icir"] for r in rows]
    mic = [r["monthly_icir"] for r in rows]
    wsc = [r["weighted_score"] for r in rows]
    imean = [r.get("icir_mean", 0.0) for r in rows]

    def _one(path: str, use_abs: bool) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        if use_abs:
            oicp, dicp, micp, wscp, imeanp = (
                np.abs(np.asarray(oic, dtype=float)),
                np.abs(np.asarray(dic, dtype=float)),
                np.abs(np.asarray(mic, dtype=float)),
                np.abs(np.asarray(wsc, dtype=float)),
                np.abs(np.asarray(imean, dtype=float)),
            )
            title = "Pool metrics (absolute values)"
        else:
            oicp, dicp, micp, wscp, imeanp = oic, dic, mic, wsc, imean
            title = "Pool metrics (raw)"

        ax.plot(steps, oicp, label="overall_ic", alpha=0.85)
        ax.plot(steps, dicp, label="daily_icir", alpha=0.85)
        ax.plot(steps, micp, label="monthly_icir", alpha=0.85)
        ax.plot(steps, imeanp, label="icir_mean", alpha=0.75)
        ax.plot(steps, wscp, label="weighted_ic_score", linestyle="--", alpha=0.75)
        ax.set_xlabel("train_step")
        ax.set_ylabel("value")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    _one(path_raw, use_abs=False)
    _one(path_abs, use_abs=True)


def update_pool_monitoring(
    step: int,
    run_dir: str,
    pool: AlphaPoolState,
    returns: pd.Series,
    train_idx: pd.Index,
    test_idx: pd.Index,
    trainer: LinearMeanStdEnsembleTrainer,
    *,
    fragment_eval: bool,
    icir_missing_gamma: float,
    icir_missing_eps: float,
    bt: AlphaBacktest,
    history: List[Dict[str, Any]],
) -> None:
    """
    用当前 pool 在 train/test 划分下拟合权重，生成全时段线性预测；
    在全样本上与 returns 对齐后，用完整 evaluate（含 compliance）与 IC 分解指标记录并落盘。
    """
    factors = pool.factor_series_list()
    pool_dir = os.path.join(run_dir, "pool_artifacts")
    os.makedirs(pool_dir, exist_ok=True)

    if not factors or len(train_idx) == 0 or len(test_idx) == 0:
        return

    fit = trainer.fit(
        factors,
        returns,
        train_idx,
        test_idx,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
        fragment_eval=fragment_eval,
    )
    pred = build_full_prediction(factors, returns, fit.weights)
    ret_a = returns.reindex(pred.index)

    pred.to_csv(os.path.join(pool_dir, "prediction_latest.csv"), header=True)
    pd.DataFrame({"prediction": pred, "returns": ret_a}).to_csv(
        os.path.join(pool_dir, "prediction_returns_latest.csv")
    )

    m = metrics_from_factor_returns_full(
        pred,
        ret_a,
        icir_missing_gamma=icir_missing_gamma,
        icir_missing_eps=icir_missing_eps,
    )
    if m is None:
        return

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

    with open(os.path.join(pool_dir, "pool_metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    raw_path = os.path.join(run_dir, "pool_ic_raw.png")
    abs_path = os.path.join(run_dir, "pool_ic_abs.png")
    _plot_ic_curves(history, raw_path, abs_path)
