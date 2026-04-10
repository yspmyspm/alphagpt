"""AlphaPool config loading and path resolution."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

ALPHAPOOL_ACTOR_NAME = "AlphaPool"
ALPHAPOOL_NAMESPACE = "alpha_services"


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_alphapool_config_file(path: str) -> Dict[str, Any]:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"AlphaPool config file not found: {resolved}")
    return _read_json(resolved)




def resolve_data_root(config_path: str, cfg: Dict[str, Any]) -> Path:
    raw = str(cfg["paths"]["data_root"]).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(config_path).resolve().parent / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_optional_path(base: Path, value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw in {"", "null", "None"}:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def resolve_persist_dir(config_path: str, cfg: Dict[str, Any]) -> Optional[str]:
    return _resolve_optional_path(
        resolve_data_root(config_path, cfg), cfg["paths"]["persist_dir"]
    )


def resolve_initial_pool_path(config_path: str, cfg: Dict[str, Any]) -> Optional[str]:
    return _resolve_optional_path(
        resolve_data_root(config_path, cfg), cfg["paths"]["initial_pool_path"]
    )