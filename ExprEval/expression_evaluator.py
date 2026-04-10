from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from configs.config import ExprEvalConfig, load_config
from data_source import FeatherColumnSource
from operator_registry import OperatorRegistry
from rpn_executor import RPNExecutor


class ExpressionEvaluator:
    """Load required columns and evaluate one postfix expression locally."""

    def __init__(
        self,
        config: ExprEvalConfig | None = None,
        config_path: str | None = None,
    ):
        self.config = config or load_config(config_path)
        self.registry = OperatorRegistry(self.config.ts_parameters)
        self.data_source = FeatherColumnSource(
            default_data_path=self.config.default_data_path,
            time_column=self.config.time_column,
        )
        self.executor = RPNExecutor(self.registry)

    def _normalize_expression(
        self,
        expression: Sequence[object] | str,
    ) -> tuple[object, ...]:
        if isinstance(expression, str):
            tokens = [token for token in expression.strip().split() if token]
        elif isinstance(expression, Iterable):
            tokens = list(expression)
        else:
            raise TypeError(f"Unsupported expression type: {type(expression)!r}")
        if not tokens:
            raise ValueError("Expression is empty")
        return tuple(tokens)

    def _required_features(self, tokens: Sequence[object]) -> tuple[str, ...]:
        return tuple(sorted({
            str(token)
            for token in tokens
            if not self.registry.is_operator(token)
            and self.registry.parse_parameter_token(token) is None
        }))

    def _analyze_expression(
        self,
        expression: Sequence[object] | str,
    ) -> dict[str, object]:
        tokens = self._normalize_expression(expression)
        text = " ".join(str(token) for token in tokens)
        expression_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        required_features = self._required_features(tokens)
        return {
            "tokens": tokens,
            "text": text,
            "expression_hash": expression_hash,
            "required_features": required_features,
        }

    def get_capabilities(self, data_path: str | None = None) -> dict[str, Any]:
        resolved_path = self.data_source.resolve_data_path(data_path)
        feature_names = self.data_source.list_features(resolved_path)
        return self.registry.build_capability(feature_names, resolved_path)

    def warmup(self, data_path: str | None = None) -> dict[str, Any]:
        resolved_path = self.data_source.resolve_data_path(data_path)
        self.data_source.list_columns(resolved_path)
        return self.get_capabilities(resolved_path)

    def evaluate_payload(
        self,
        expression: list[object] | tuple[object, ...] | str,
        data_path: str | None = None,
    ) -> dict[str, Any]:
        resolved_path = self.data_source.resolve_data_path(data_path)
        expr = self._analyze_expression(expression)
        frame = self.data_source.load_frame(expr["required_features"], resolved_path)
        series = self.executor.execute(expr["tokens"], frame)
        return {
            "data_path": resolved_path,
            "expression": expr["text"],
            "required_features": list(expr["required_features"]),
            "expression_hash": expr["expression_hash"],
            "series": series,
        }

    def ping(self) -> str:
        return "ok"
