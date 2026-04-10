from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPR_EVAL_ACTOR_NAME = "ExprEval"
EXPR_EVAL_NAMESPACE = "alpha_services"


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    raw = path if path is not None else os.environ.get("EXPR_EVAL_CONFIG")
    if raw is None or not str(raw).strip():
        raw = Path(__file__).resolve().parent / "config.json"
    return Path(raw).expanduser().resolve()


def load_raw_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"ExprEval config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_optional_path(config_path: Path, value: Any) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    else:
        path = path.resolve()
    return str(path)


@dataclass(frozen=True)
class ExprEvalConfig:
    default_data_path: str | None
    time_column: str
    ts_parameters: list[int]


def load_config(path: str | os.PathLike[str] | None = None) -> ExprEvalConfig:
    resolved = resolve_config_path(path)
    raw = load_raw_config(resolved)
    data_cfg = raw["data"]
    op_cfg = raw["operators"]
    return ExprEvalConfig(
        default_data_path=_resolve_optional_path(resolved, data_cfg["default_data_path"]),
        time_column=str(data_cfg["time_column"]),
        ts_parameters=[int(v) for v in op_cfg["ts_parameters"]],
    )


def load_deployment_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    raw = load_raw_config(resolved)
    deploy = raw["deployment"]
    memory = deploy["memory"]
    resources = deploy["resources"]
    label_selector = deploy["label_selector"]
    return {
        "ray_address": deploy["ray_address"],
        "actor_name": EXPR_EVAL_ACTOR_NAME,
        "namespace": EXPR_EVAL_NAMESPACE,
        "worker_count": int(deploy["worker_count"]),
        "worker_num_cpus": int(deploy["worker_num_cpus"]),
        "worker_memory": None if memory in (None, "") else int(memory),
        "worker_resources": dict(resources) if isinstance(resources, dict) else None,
        "worker_label_selector": (
            dict(label_selector) if isinstance(label_selector, dict) and label_selector else None
        ),
        "replace_existing": bool(deploy["replace_existing"]),
        "keep_alive": bool(deploy["keep_alive"]),
    }
