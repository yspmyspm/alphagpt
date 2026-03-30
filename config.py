"""Backward-compatibility shim – all config logic lives in configs.config."""

from configs.config import ModelConfig, install_config, load_merged, write_run_config_snapshot

__all__ = [
    "ModelConfig",
    "install_config",
    "load_merged",
    "write_run_config_snapshot",
]

