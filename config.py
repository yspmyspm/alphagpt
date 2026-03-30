"""
训练/数据配置：默认从项目根目录的 config.json 读取，可用环境变量 CONFIG_PATH 指定其他路径。
config.json 按类别分块（runtime / data / training / …），顶层也可写扁平键覆盖（兼容旧格式）。
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch

# 最近一次合并后的配置（用于写入每次 run 的 config_snapshot.json）
_MERGED_AT_LOAD: Optional[Dict[str, Any]] = None


def _config_file_path() -> str:
    return os.environ.get(
        "CONFIG_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
    )


def _defaults_nested() -> dict:
    """分类默认项；eval_num_workers 为 0 时表示「自动」：在 install_config 里解析。"""
    return {
        "runtime": {
            "device": "auto",
            "eval_num_workers": 0,
            "logging_dir": "runs",
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
        },
        "model": {
            "max_formula_len": 12,
            "gen_temperature": 1.0,
            "ts_parameters": [1, 5, 10, 30, 60, 120, 1440],
        },
        "scoring": {
            "use_smooth_reward": True,
            "backtest_penalty": -5.0,
            "unfinished_penalty": -5.0,
            "icir_missing_gamma": 2.0,
            "icir_missing_eps": 1e-6,
        },
        "alpha_pool": {
            "use_alpha_pool": True,
            "alpha_pool_size": 32,
            "alpha_train_years": 2.0,
            "alpha_missing_threshold": 0.3,
            "alpha_pool_fragment_eval": True,
            "ensemble_maxiter": 400,
        },
    }


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
    单层分类 dict -> 扁平 dict。子表中的键先展开，顶层非 dict 键最后写入（可覆盖同名扁平键，兼容旧版顶层 batch_size 等）。
    """
    flat: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            flat.update(v)
    for k, v in obj.items():
        if not isinstance(v, dict):
            flat[k] = v
    return flat


def load_merged() -> dict:
    """合并默认项与 config.json（后者覆盖前者），返回扁平 dict。"""
    merged = deepcopy(_defaults_nested())
    path = _config_file_path()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        merged = _deep_merge(merged, user)
    return _flatten_config(merged)


def _device_from_str(s: str | None) -> torch.device:
    if s is None or s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def install_config() -> None:
    """将合并后的配置写入 ModelConfig。"""
    global _MERGED_AT_LOAD
    m = load_merged()
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
    ModelConfig.MAX_FORMULA_LEN = int(m["max_formula_len"])
    ModelConfig.GEN_TEMPERATURE = float(m["gen_temperature"])
    ModelConfig.TS_PARAMETERS = [int(x) for x in m["ts_parameters"]]

    ModelConfig.USE_SMOOTH_REWARD = bool(m["use_smooth_reward"])
    ModelConfig.BACKTEST_PENALTY = float(m["backtest_penalty"])
    ModelConfig.UNFINISHED_PENALTY = float(m["unfinished_penalty"])
    ModelConfig.ICIR_MISSING_GAMMA = float(m["icir_missing_gamma"])
    ModelConfig.ICIR_MISSING_EPS = float(m["icir_missing_eps"])

    ew = int(m["eval_num_workers"])
    ModelConfig.EVAL_NUM_WORKERS = (
        max(1, (os.cpu_count() or 4) - 1) if ew <= 0 else ew
    )
    ModelConfig.LOGGING_DIR = m["logging_dir"]
    ModelConfig.SAVE_CHECKPOINT_EVERY = int(m["save_checkpoint_every"])

    ModelConfig.USE_ALPHA_POOL = bool(m.get("use_alpha_pool", True))
    ModelConfig.ALPHA_POOL_SIZE = int(m.get("alpha_pool_size", 32))
    ModelConfig.ALPHA_TRAIN_YEARS = float(m.get("alpha_train_years", 2.0))
    ModelConfig.ALPHA_MISSING_THRESHOLD = float(m.get("alpha_missing_threshold", 0.3))
    ModelConfig.ALPHA_POOL_FRAGMENT_EVAL = bool(m.get("alpha_pool_fragment_eval", True))
    ModelConfig.ENSEMBLE_MAXITER = int(m.get("ensemble_maxiter", 400))

    ModelConfig.INPUT_DIM = None


class ModelConfig:
    """
    由 install_config() 填充；训练过程中 data_loader 会设置 INPUT_DIM。
    """

    pass


def write_run_config_snapshot(run_dir: str) -> None:
    """将本次生效的配置（含解析后的 device、运行时的 input_dim）写入 run 目录。"""
    snap = dict(_MERGED_AT_LOAD or load_merged())
    snap["input_dim"] = ModelConfig.INPUT_DIM
    snap["device_resolved"] = str(ModelConfig.DEVICE)
    path = os.path.join(run_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, default=str)


install_config()
