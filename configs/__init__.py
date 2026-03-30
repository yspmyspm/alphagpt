"""对外统一入口：``from configs import ModelConfig, install_config, ...``。"""

from .config import ModelConfig, install_config, load_merged, write_run_config_snapshot
from .config import __all__ as __all__
