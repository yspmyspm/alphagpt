from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch

_NESTED_AT_LOAD: Optional[Dict[str, Any]] = None
_CONFIG_PATH_AT_LOAD: Optional[str] = None


def _resolve_config_path(path: Optional[str]) -> str:
    raw = path if path is not None else os.environ.get("CONFIG_PATH")
    if raw is None or not str(raw).strip():
        raise ValueError(
            "config path is required (set --config or CONFIG_PATH)"
        )
    return os.path.abspath(os.path.expanduser(str(raw).strip()))


def _resolve_path_from_config(config_path: str, value: Any) -> Any:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if os.path.isabs(s):
        return os.path.abspath(os.path.expanduser(s))
    base_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.abspath(os.path.join(base_dir, s))


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _device_from_str(s: str | None) -> torch.device:
    if s is None or s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def install_config(config_path: Optional[str] = None) -> None:
    global _NESTED_AT_LOAD, _CONFIG_PATH_AT_LOAD
    resolved_path = _resolve_config_path(config_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"main config file not found: {resolved_path}")
    os.environ["CONFIG_PATH"] = resolved_path
    _CONFIG_PATH_AT_LOAD = resolved_path

    cfg = _read_json(resolved_path)
    _NESTED_AT_LOAD = cfg

    rt = cfg["runtime"]
    ModelConfig.DEVICE = _device_from_str(rt["device"])
    ModelConfig.LOGGING_DIR = _resolve_path_from_config(resolved_path, rt["logging_dir"])

    ModelConfig.EXPR_DATA_PATH = _resolve_path_from_config(
        resolved_path, cfg["ExprEval"]["data_path"]
    )

    tr = cfg["training"]
    ModelConfig.BATCH_SIZE = int(tr["batch_size"])
    ModelConfig.TRAIN_STEPS = int(tr["train_steps"])
    ModelConfig.SAVE_CHECKPOINT_EVERY = int(tr["save_checkpoint_every"])
    ModelConfig.OPTIMIZER_LR = float(tr["optimizer_lr"])
    ModelConfig.POLICY_ADVANTAGE_EPS = float(tr["policy_advantage_eps"])

    res = cfg["resume"]
    ModelConfig.RESUME_MODEL_CHECKPOINT = _resolve_path_from_config(
        resolved_path, res["resume_model_checkpoint"]
    )
    ModelConfig.RESUME_ALPHA_POOL_PATH = _resolve_path_from_config(
        resolved_path, res["resume_alpha_pool_path"]
    )

    md = cfg["model"]
    ModelConfig.MAX_FORMULA_LEN = int(md["max_formula_len"])
    ModelConfig.GEN_TEMPERATURE = float(md["gen_temperature"])
    ModelConfig.SAMPLING_TEMPERATURE_FLOOR = float(md["sampling_temperature_floor"])
    ModelConfig.MAX_DECODE_STACK_SIZE = int(md["max_decode_stack_size"])

    arch = cfg["model_architecture"]
    ModelConfig.MODEL_D_MODEL = int(arch["d_model"])
    ModelConfig.MODEL_NHEAD = int(arch["nhead"])
    ModelConfig.MODEL_NUM_LAYERS = int(arch["num_layers"])
    ModelConfig.MODEL_DIM_FEEDFORWARD = int(arch["dim_feedforward"])
    ModelConfig.MODEL_NUM_LOOPS = int(arch["num_loops"])
    ModelConfig.MODEL_DROPOUT = float(arch["dropout"])
    ModelConfig.MODEL_MTP_NUM_TASKS = int(arch["mtp_num_tasks"])

    reg = cfg["regularization"]
    ModelConfig.USE_LORD_REGULARIZATION = bool(reg["use_lord_regularization"])
    ModelConfig.LORD_DECAY_RATE = float(reg["lord_decay_rate"])
    ModelConfig.LORD_NUM_ITERATIONS = int(reg["lord_num_iterations"])
    ModelConfig.LORD_DECAY_KEYWORDS = list(reg["lord_decay_keywords"])
    ModelConfig.LORD_RANK_MONITOR_KEYWORDS = list(reg["lord_rank_monitor_keywords"])
    ModelConfig.LORD_RANK_LOG_EVERY = int(reg["lord_rank_log_every"])

    sc = cfg["scoring"]
    ModelConfig.NO_EOS_PENALTY = float(sc["no_eos_penalty"])
    ModelConfig.REWARD_W_DAILY = float(sc["reward_w_daily"])
    ModelConfig.REWARD_W_MONTHLY = float(sc["reward_w_monthly"])
    ModelConfig.REWARD_W_IC = float(sc["reward_w_ic"])
    ModelConfig.USE_ALPHA_POOL = bool(cfg["alpha_pool"]["use_alpha_pool"])

    ModelConfig.INPUT_DIM = None


class ModelConfig:
    POLICY_ADVANTAGE_EPS: float
    SAMPLING_TEMPERATURE_FLOOR: float
    REWARD_W_DAILY: float
    REWARD_W_MONTHLY: float
    REWARD_W_IC: float
    INPUT_DIM: int | None = None


def write_run_config_snapshot(run_dir: str) -> None:
    snap = deepcopy(_NESTED_AT_LOAD or _read_json(_resolve_config_path(None)))
    snap["input_dim"] = ModelConfig.INPUT_DIM
    snap["device_resolved"] = str(ModelConfig.DEVICE)
    snap["config_path"] = _CONFIG_PATH_AT_LOAD
    path = os.path.join(run_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, default=str)


__all__ = [
    "ModelConfig",
    "install_config",
    "write_run_config_snapshot",
]
