"""
可 pickle 的评估 worker，供 ProcessPoolExecutor 使用。
独立模块避免 Windows spawn 时重复执行主脚本。
每个 formula 单独提交一次，通过 initializer 在 worker 内预加载数据。
"""
from factors import FeatureEngineer
from vm import StackVM
from backtest import AlphaBacktest
from helpers.reward_metrics import compute_compliance_only, missing_ratio_on_common

# 每个 worker 进程内的全局数据，由 _init_worker 设置
_worker_features = None
_worker_returns = None
_worker_vm = None
_worker_bt = None

# alphapool 并行阶段
_worker_missing_threshold = 0.3
_worker_use_smooth = True


def _init_worker(features, returns, input_dim, use_smooth_reward):
    """每个 worker 启动时调用一次，预加载数据避免重复序列化。"""
    global _worker_features, _worker_returns, _worker_vm, _worker_bt
    _worker_features = features
    _worker_returns = returns
    FeatureEngineer.INPUT_DIM = input_dim
    _worker_vm = StackVM()
    _worker_bt = AlphaBacktest(use_smooth_reward=use_smooth_reward)


def _init_worker_pool(features, returns, input_dim, use_smooth_reward, missing_threshold):
    """alphapool：仅 VM + 数据 + 缺失阈值；合规在 reward_metrics 中算。"""
    global _worker_features, _worker_returns, _worker_vm, _worker_missing_threshold, _worker_use_smooth
    _worker_features = features
    _worker_returns = returns
    FeatureEngineer.INPUT_DIM = input_dim
    _worker_vm = StackVM()
    _worker_missing_threshold = float(missing_threshold)
    _worker_use_smooth = bool(use_smooth_reward)


def eval_single_formula(formula):
    """
    评估单个公式。依赖 _init_worker 预加载的数据。
    returns: (reward, score, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula)
    """
    res = _worker_vm.execute(formula, _worker_features)
    if res is None:
        return (-5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, formula)
    std_val = res.std()
    if std_val < 1e-4:
        ratio = min(1.0, std_val / 1e-4)
        reward = -2.0 * (1.0 - ratio)
        return (reward, reward, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, formula)
    out = _worker_bt.evaluate(res, _worker_returns)
    reward = float(out[1])
    daily_icir = out[2]
    monthly_icir = out[3]
    overall_ic = out[4]
    s_finite = out[5] if len(out) > 5 else 0.0
    s_dist = out[6] if len(out) > 6 else 0.0
    s_halflife = out[7] if len(out) > 7 else 0.0
    compliance = out[8] if len(out) > 8 else 0.0
    return (reward, reward, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula)


def eval_formula_for_pool(formula):
    """
    alphapool 并行阶段：执行公式、缺失率、低方差、合规。
    返回 dict，供 AlphaPoolBatchEvaluator 在主进程算 reward / 更新 pool。
    """
    res = _worker_vm.execute(formula, _worker_features)
    if res is None:
        return {"ok": False, "reason": "execute", "formula": formula}


    mr = missing_ratio_on_common(res, _worker_returns)
    if mr > _worker_missing_threshold:
        return {"ok": True, "missing_high": True, "missing_ratio": mr, "formula": formula}

    std_val = res.std()

    if std_val < 1e-4:
        ratio = min(1.0, std_val / 1e-4)
        reward = -2.0 * (1.0 - ratio)
        return {"ok": True, "low_std": True, "reward": reward, "formula": formula}
    



    comp = compute_compliance_only(res, _worker_returns, use_smooth_reward=_worker_use_smooth)
    
    if comp is None:
        return {"ok": False, "reason": "compliance", "formula": formula}
    

    s_fin, s_dist, s_halflife, compliance = comp
    return {
        "ok": True,
        "factor": res,
        "missing_ratio": mr,
        "compliance": compliance,
        "s_finite": s_fin,
        "s_dist": s_dist,
        "s_halflife": s_halflife,
        "formula": formula,
    }
