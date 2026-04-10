from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from operator_registry import OperatorRegistry


class RPNExecutor:
    """Pure stack executor for postfix expression tokens."""

    def __init__(self, registry: OperatorRegistry):
        self.registry = registry

    def execute(self, tokens: Sequence[object], frame: pd.DataFrame) -> pd.Series:
        stack: list[pd.Series | int] = []
        feature_names = set(frame.columns)

        for token in tokens:
            operator_entry = self.registry.get_operator(token)
            if operator_entry is not None:
                stack.append(self._apply_operator(operator_entry, stack))
                continue

            param = self.registry.parse_parameter_token(token)
            if param is not None:
                stack.append(param)
                continue

            feature_name = str(token)
            if feature_name not in feature_names:
                raise ValueError(f"Unknown feature token: {feature_name}")
            stack.append(frame[feature_name].astype(float).copy())

        if len(stack) != 1:
            raise ValueError(f"Expression left {len(stack)} values on stack: {list(tokens)}")

        result = stack[0]
        if not isinstance(result, pd.Series):
            raise ValueError(f"Final stack value is not a pandas Series: {list(tokens)}")
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result.name = "expr"
        return result

    def _apply_operator(self, operator_entry, stack: list[pd.Series | int]) -> pd.Series:
        spec, func = operator_entry
        if spec.kind == "binary":
            if len(stack) < 2:
                raise ValueError(f"Operator {spec.name} requires 2 operands")
            y = stack.pop()
            x = stack.pop()
            self._assert_series(x, spec.name)
            self._assert_series(y, spec.name)
            result = func(x, y)
        elif spec.kind in {"unary_parameterized", "ts_unary"}:
            if len(stack) < 2:
                raise ValueError(f"Operator {spec.name} requires 1 operand and 1 parameter")
            param = stack.pop()
            x = stack.pop()
            self._assert_series(x, spec.name)
            self._assert_parameter(param, spec.name)
            result = func(x, int(param))
        elif spec.kind == "unary_parameterless":
            if len(stack) < 1:
                raise ValueError(f"Operator {spec.name} requires 1 operand")
            x = stack.pop()
            self._assert_series(x, spec.name)
            result = func(x)
        elif spec.kind == "ts_binary":
            if len(stack) < 3:
                raise ValueError(f"Operator {spec.name} requires 2 operands and 1 parameter")
            param = stack.pop()
            y = stack.pop()
            x = stack.pop()
            self._assert_series(x, spec.name)
            self._assert_series(y, spec.name)
            self._assert_parameter(param, spec.name)
            result = func(x, y, int(param))
        else:
            raise ValueError(f"Unsupported operator kind: {spec.kind}")

        if result is None or not isinstance(result, pd.Series):
            raise ValueError(f"Operator {spec.name} returned invalid result")
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _assert_series(value: object, operator_name: str) -> None:
        if not isinstance(value, pd.Series):
            raise ValueError(f"Operator {operator_name} expected a pandas Series")

    @staticmethod
    def _assert_parameter(value: object, operator_name: str) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Operator {operator_name} expected an integer parameter")
