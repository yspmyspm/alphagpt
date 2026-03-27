"""
训练/数据配置：默认从项目根目录的 config.json 读取，可用环境变量 CONFIG_PATH 指定其他路径。
修改参数请编辑 config.json，无需改本文件。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch

# 最近一次合并后的配置（用于写入每次 run 的 config_snapshot.json）
_MERGED_AT_LOAD: Optional[Dict[str, Any]] = None


def _config_file_path() -> str:
    return os.environ.get(
        "CONFIG_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
    )


def _defaults() -> dict:
    return {
        "device": "auto",
        "feather_path": os.path.join("data", "olhcv.feather"),
        "returns_dir": "returns",
        "returns_filename": "30min.feather",
        "returns_column": "returns",
        "feature_columns": None,
        "batch_size": 1024,
        "train_steps": 1000,
        "max_formula_len": 12,
        "gen_temperature": 1.0,
        "ts_parameters": [1, 5, 10, 30, 60, 120, 1440],
        "use_smooth_reward": True,
        "backtest_penalty": -5.0,
        "unfinished_penalty": -5.0,
        "icir_missing_gamma": 2.0,
        "icir_missing_eps": 1e-6,
        "eval_num_workers": max(1, (os.cpu_count() or 4) - 1),
        "logging_dir": "runs",
        "save_checkpoint_every": 100,
    }


def load_merged() -> dict:
    """合并默认项与 config.json（后者覆盖前者）。"""
    merged = _defaults().copy()
    path = _config_file_path()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        merged.update(user)
    return merged


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
