"""
训练与数据配置（唯一实现模块）。

- 对外使用 ``from configs import ModelConfig, install_config, ...``（或 ``from configs.config import ...``）。
- 默认读取与本模块同目录的 ``configs/config.json``；也可用环境变量 ``CONFIG_PATH`` 或 ``install_config(path)`` / ``python engine.py --config ...`` 覆盖。
- JSON 支持 ``sub_config_paths``（字符串列表或字典值列表），先合并子文件再与主文件合并，最后与内置默认值深度合并。
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Set

import torch

_MERGED_AT_LOAD: Optional[Dict[str, Any]] = None
_CONFIG_PATH_AT_LOAD: Optional[str] = None


def _default_config_file_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _resolve_config_path(path: Optional[str]) -> str:
    raw = path if path is not None else os.environ.get("CONFIG_PATH")
    if raw is None or not str(raw).strip():
        raw = _default_config_file_path()
    return os.path.abspath(os.path.expanduser(str(raw).strip()))


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v) if isinstance(v, dict) else v
    return out


def _flatten_config(obj: dict) -> dict:
    """
    One-level categorized dict -> flattened dict.
    Child dict keys are expanded first; top-level non-dict keys are written last
    so legacy flat keys can still override categorized values.
    """
    flat: Dict[str, Any] = {}
    for _, v in obj.items():
        if isinstance(v, dict):
            flat.update(v)
    for k, v in obj.items():
        if not isinstance(v, dict):
            flat[k] = v
    return flat


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def _iter_subconfig_paths(sub_cfg: Any) -> list[str]:
    if sub_cfg is None:
        return []
    if isinstance(sub_cfg, dict):
        return [str(v) for v in sub_cfg.values()]
    if isinstance(sub_cfg, list):
        return [str(v) for v in sub_cfg]
    raise ValueError("sub_config_paths must be a dict or list")


def _load_user_with_subconfigs(path: str, visited: Optional[Set[str]] = None) -> dict:
    resolved = os.path.abspath(os.path.expanduser(path))
    if visited is None:
        visited = set()
    if resolved in visited:
        raise ValueError(f"Cyclic sub-config include detected: {resolved}")
    visited.add(resolved)

    cur = _read_json(resolved)
    sub_paths = _iter_subconfig_paths(cur.pop("sub_config_paths", None))

    merged_sub: dict = {}
    base_dir = os.path.dirname(resolved)
    for rel in sub_paths:
        sub_resolved = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if not os.path.isfile(sub_resolved):
            raise FileNotFoundError(f"sub config not found: {sub_resolved}")
        sub_obj = _load_user_with_subconfigs(sub_resolved, visited=visited)
        merged_sub = _deep_merge(merged_sub, sub_obj)

    visited.remove(resolved)
    return _deep_merge(merged_sub, cur)


def _defaults_nested() -> dict:
    return {
        "runtime": {
            "device": "auto",
            "eval_num_workers": 0,
            "logging_dir": "runs",
            "mp_start_method": "spawn",
        },
        "data": {
            "feather_path": os.path.join("data", "olhcv.feather"),
            "returns_dir": "returns",
            "returns_filename": "30min.feather",
            "returns_column": "returns",
            "feature_columns": None,
        },
        "training": {
            "batch_size": 1024,
            "train_steps": 1000,
            "save_checkpoint_every": 100,
            "optimizer_lr": 1e-3,
        },
        "resume": {
            "resume_model_checkpoint": None,
            "resume_alpha_pool_path": None,
        },
        "model": {
            "max_formula_len": 12,
            "gen_temperature": 1.0,
            "ts_parameters": [1, 5, 10, 30, 60, 120, 1440],
            "max_decode_stack_size": 16,
        },
        "model_architecture": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "num_loops": 3,
            "dropout": 0.1,
            "mtp_num_tasks": 3,
        },
        "compliance": {
            "distribution_check_skew_limit": 10.0,
            "distribution_check_kurt_limit": 100.0,
            "score_finite_min_ratio": 0.95,
            "score_distribution_skew_limit": 40.0,
            "score_distribution_kurt_limit": 1000.0,
            "score_halflife_min_corr": 0.5,
            "halflife_lag": 5,
        },
        "regularization": {
            "use_lord_regularization": False,
            "lord_decay_rate": 1e-3,
            "lord_num_iterations": 5,
            "lord_decay_keywords": ["q_proj", "k_proj", "attention", "qk_norm"],
            "lord_rank_monitor_keywords": ["q_proj", "k_proj"],
            "lord_rank_log_every": 100,
        },
        "scoring": {
            "use_smooth_reward": True,
            "backtest_penalty": -5.0,
            "no_eos_penalty": -5.0,
            "execute_fail_penalty": -5.0,
            "missing_high_penalty": -5.0,
            "low_std_penalty_base": -2.0,
            "low_std_threshold": 1e-4,
            "compliance_fail_penalty": -5.0,
            "icir_missing_gamma": 2.0,
            "icir_missing_eps": 1e-6,
            "reward_weight_daily_icir": 0.3,
            "reward_weight_monthly_icir": 0.3,
            "reward_weight_ic_score": 0.4,
        },
        "alpha_pool": {
            "use_alpha_pool": True,
            "alpha_pool_size": 32,
            "alpha_missing_threshold": 0.3,
            "alpha_max_corr_min_points": 10,
            "ensemble_maxiter": 400,
        },
    }


def load_merged(config_path: Optional[str] = None) -> dict:
    merged = deepcopy(_defaults_nested())
    resolved_path = _resolve_config_path(config_path)
    if os.path.isfile(resolved_path):
        user = _load_user_with_subconfigs(resolved_path)
        merged = _deep_merge(merged, user)
    return _flatten_config(merged)


def _device_from_str(s: str | None) -> torch.device:
    if s is None or s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def install_config(config_path: Optional[str] = None) -> None:
    global _MERGED_AT_LOAD, _CONFIG_PATH_AT_LOAD
    resolved_path = _resolve_config_path(config_path)
    os.environ["CONFIG_PATH"] = resolved_path
    _CONFIG_PATH_AT_LOAD = resolved_path

    m = load_merged(resolved_path)
    _MERGED_AT_LOAD = m

    ModelConfig.DEVICE = _device_from_str(m.get("device", "auto"))
    ModelConfig.FEATHER_PATH = m["feather_path"]
    ModelConfig.RETURNS_DIR = m["returns_dir"]
    ModelConfig.RETURNS_FILENAME = m["returns_filename"]
    ModelConfig.RETURNS_COLUMN = m["returns_column"]

    fc = m.get("feature_columns")
    ModelConfig.FEATURE_COLUMNS = None if fc is None else list(fc)

    ModelConfig.BATCH_SIZE = int(m["batch_size"])
    ModelConfig.TRAIN_STEPS = int(m["train_steps"])
    ModelConfig.SAVE_CHECKPOINT_EVERY = int(m["save_checkpoint_every"])
    ModelConfig.OPTIMIZER_LR = float(m["optimizer_lr"])

    ModelConfig.MAX_FORMULA_LEN = int(m["max_formula_len"])
    ModelConfig.GEN_TEMPERATURE = float(m["gen_temperature"])
    ModelConfig.TS_PARAMETERS = [int(x) for x in m["ts_parameters"]]
    ModelConfig.MAX_DECODE_STACK_SIZE = int(m["max_decode_stack_size"])

    ModelConfig.MODEL_D_MODEL = int(m["d_model"])
    ModelConfig.MODEL_NHEAD = int(m["nhead"])
    ModelConfig.MODEL_NUM_LAYERS = int(m["num_layers"])
    ModelConfig.MODEL_DIM_FEEDFORWARD = int(m["dim_feedforward"])
    ModelConfig.MODEL_NUM_LOOPS = int(m["num_loops"])
    ModelConfig.MODEL_DROPOUT = float(m["dropout"])
    ModelConfig.MODEL_MTP_NUM_TASKS = int(m["mtp_num_tasks"])

    ModelConfig.DISTRIBUTION_CHECK_SKEW_LIMIT = float(m["distribution_check_skew_limit"])
    ModelConfig.DISTRIBUTION_CHECK_KURT_LIMIT = float(m["distribution_check_kurt_limit"])
    ModelConfig.SCORE_FINITE_MIN_RATIO = float(m["score_finite_min_ratio"])
    ModelConfig.SCORE_DISTRIBUTION_SKEW_LIMIT = float(m["score_distribution_skew_limit"])
    ModelConfig.SCORE_DISTRIBUTION_KURT_LIMIT = float(m["score_distribution_kurt_limit"])
    ModelConfig.SCORE_HALFLIFE_MIN_CORR = float(m["score_halflife_min_corr"])
    ModelConfig.HALFLIFE_LAG = int(m["halflife_lag"])

    ModelConfig.USE_LORD_REGULARIZATION = bool(m["use_lord_regularization"])
    ModelConfig.LORD_DECAY_RATE = float(m["lord_decay_rate"])
    ModelConfig.LORD_NUM_ITERATIONS = int(m["lord_num_iterations"])
    ModelConfig.LORD_DECAY_KEYWORDS = list(m["lord_decay_keywords"])
    ModelConfig.LORD_RANK_MONITOR_KEYWORDS = list(m["lord_rank_monitor_keywords"])
    ModelConfig.LORD_RANK_LOG_EVERY = int(m["lord_rank_log_every"])

    ModelConfig.USE_SMOOTH_REWARD = bool(m["use_smooth_reward"])
    ModelConfig.BACKTEST_PENALTY = float(m["backtest_penalty"])
    ModelConfig.NO_EOS_PENALTY = float(m["no_eos_penalty"])
    ModelConfig.EXECUTE_FAIL_PENALTY = float(m["execute_fail_penalty"])
    ModelConfig.MISSING_HIGH_PENALTY = float(m["missing_high_penalty"])
    ModelConfig.LOW_STD_PENALTY_BASE = float(m["low_std_penalty_base"])
    ModelConfig.LOW_STD_THRESHOLD = float(m["low_std_threshold"])
    ModelConfig.COMPLIANCE_FAIL_PENALTY = float(m["compliance_fail_penalty"])
    ModelConfig.ICIR_MISSING_GAMMA = float(m["icir_missing_gamma"])
    ModelConfig.ICIR_MISSING_EPS = float(m["icir_missing_eps"])
    ModelConfig.REWARD_WEIGHT_DAILY_ICIR = float(m["reward_weight_daily_icir"])
    ModelConfig.REWARD_WEIGHT_MONTHLY_ICIR = float(m["reward_weight_monthly_icir"])
    ModelConfig.REWARD_WEIGHT_IC_SCORE = float(m["reward_weight_ic_score"])

    ew = int(m["eval_num_workers"])
    ModelConfig.EVAL_NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1) if ew <= 0 else ew
    ModelConfig.LOGGING_DIR = m["logging_dir"]
    ModelConfig.MP_START_METHOD = str(m.get("mp_start_method", "spawn"))

    ModelConfig.RESUME_MODEL_CHECKPOINT = m.get("resume_model_checkpoint") or None
    ModelConfig.RESUME_ALPHA_POOL_PATH = m.get("resume_alpha_pool_path") or None

    ModelConfig.USE_ALPHA_POOL = bool(m["use_alpha_pool"])
    ModelConfig.ALPHA_POOL_SIZE = int(m["alpha_pool_size"])
    ModelConfig.ALPHA_MISSING_THRESHOLD = float(m["alpha_missing_threshold"])
    ModelConfig.ALPHA_MAX_CORR_MIN_POINTS = int(m["alpha_max_corr_min_points"])
    ModelConfig.ENSEMBLE_MAXITER = int(m["ensemble_maxiter"])

    ModelConfig.INPUT_DIM = None


class ModelConfig:
    """Dynamically populated by install_config().

    Numerical stability constants that are implementation details (not tunable)
    are defined as class-level defaults below.
    """

    POLICY_ADVANTAGE_EPS: float = 1e-5
    SAMPLING_TEMPERATURE_FLOOR: float = 1e-6
    ICIR_STD_EPS: float = 1e-8
    GEOM_STD_EPS: float = 1e-16
    IC_SCORE_DIV_EPS: float = 1e-8


def write_run_config_snapshot(run_dir: str) -> None:
    snap = dict(_MERGED_AT_LOAD or load_merged())
    snap["input_dim"] = ModelConfig.INPUT_DIM
    snap["device_resolved"] = str(ModelConfig.DEVICE)
    snap["config_path"] = _CONFIG_PATH_AT_LOAD
    path = os.path.join(run_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, default=str)


__all__ = [
    "ModelConfig",
    "install_config",
    "load_merged",
    "write_run_config_snapshot",
]

# 导入本模块即安装默认配置；无默认 JSON 时用内置默认值。
install_config()
