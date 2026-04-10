from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any

import ray

_MAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MAIN_ROOT not in sys.path:
    sys.path.insert(0, _MAIN_ROOT)

from configs.config import ModelConfig

SERVICE_NAMESPACE = "alpha_services"
EXPR_EVAL_ACTOR_NAME = "ExprEval"
ALPHAPOOL_ACTOR_NAME = "AlphaPool"


@dataclass
class RayServiceBundle:
    expr_actor: Any
    pool_actor: Any
    capabilities: dict[str, Any]
    data_path: str
    expr_data_path_overridden: bool

    @classmethod
    def connect(cls) -> "RayServiceBundle":
        if not ray.is_initialized():
            ray.init(address="auto")

        expr_actor = ray.get_actor(EXPR_EVAL_ACTOR_NAME, namespace=SERVICE_NAMESPACE)
        pool_actor = ray.get_actor(ALPHAPOOL_ACTOR_NAME, namespace=SERVICE_NAMESPACE)
        requested_data_path = ModelConfig.EXPR_DATA_PATH
        if requested_data_path:
            capabilities = ray.get(
                expr_actor.get_capabilities.remote(data_path=requested_data_path)
            )
        else:
            capabilities = ray.get(expr_actor.get_capabilities.remote())

        data_path = capabilities.get("data_path") or requested_data_path
        if not data_path:
            raise ValueError(
                "ExprEval must expose a default data path, or "
                "main.ExprEval.data_path must be configured"
            )
        return cls(
            expr_actor=expr_actor,
            pool_actor=pool_actor,
            capabilities=capabilities,
            data_path=str(data_path),
            expr_data_path_overridden=bool(requested_data_path),
        )
