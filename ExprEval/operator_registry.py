"""
Operator Registry
=================

Provides a catalogue of mathematical and time-series operators used to
evaluate postfix (RPN) alpha expressions.

Each operator is a plain static method living in a *group class*
(e.g. ``BinaryOperators``, ``TimeSeriesUnaryOperators``).  The registry
introspects these classes at init time, wraps every method into an
``OperatorSpec`` (name + kind + arity), and exposes fast lookup by
operator name so the evaluator can dispatch calls during stack-based
evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# from numba_ops import TimeSeriesNumbaOps


@dataclass(frozen=True)
class OperatorSpec:
    """Immutable descriptor for a single operator: its token name, kind
    (e.g. ``"binary"``, ``"ts_unary"``), and arity (number of stack
    items consumed, including any window-size parameter)."""

    name: str
    kind: str
    arity: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "arity": self.arity}


# ---------------------------------------------------------------------------
# Time-series unary operators (series + window parameter)
# ---------------------------------------------------------------------------

class TimeSeriesUnaryOperators:
    @staticmethod
    def ts_lag(x: pd.Series, window: int = 1) -> pd.Series:
        return x.shift(window)

    @staticmethod
    def ts_delta(x: pd.Series, window: int = 1) -> pd.Series:
        return x.diff(window)

    @staticmethod
    def ts_change_ratio(x: pd.Series, window: int = 10) -> pd.Series:
        return x.diff(window) / x.shift(window)

    @staticmethod
    def ts_sum(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).sum()

    @staticmethod
    def ts_mean(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).mean()

    @staticmethod
    def ts_std(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).std()

    @staticmethod
    def ts_min(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).min()

    @staticmethod
    def ts_max(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).max()

    @staticmethod
    def ts_range(x: pd.Series, window: int = 10) -> pd.Series:
        roll = x.rolling(window, min_periods=1)
        return roll.max() - roll.min()

    @staticmethod
    def ts_norm(x: pd.Series, window: int = 10) -> pd.Series:
        roll = x.rolling(window, min_periods=1)
        minimum = roll.min()
        maximum = roll.max()
        return (x - minimum) / (maximum - minimum)

    @staticmethod
    def ts_median(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).median()

    @staticmethod
    def ts_zscore(x: pd.Series, window: int = 10) -> pd.Series:
        roll = x.rolling(window, min_periods=1)
        return (x - roll.mean()) / roll.std()

    @staticmethod
    def ts_autocorr(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).corr(x.shift(1))

    @staticmethod
    def ts_demean(x: pd.Series, window: int = 10) -> pd.Series:
        return x - x.rolling(window, min_periods=1).mean()

    @staticmethod
    def ts_std_dev_diff(x: pd.Series, window: int = 10) -> pd.Series:
        return x.rolling(window, min_periods=1).std().diff()

    # @staticmethod
    # def ts_skew(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_skew(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_kurt(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_kurt(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_argmin(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_argmin(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_argmax(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_argmax(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_mad(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_mad(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_softmax(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_softmax(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_rank(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_rank(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_wmean(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_wmean(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_beta(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_reg_beta(x.to_numpy(), window), index=x.index)

    # @staticmethod
    # def ts_resid(x: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_reg_resid(x.to_numpy(), window), index=x.index)



# ---------------------------------------------------------------------------
# Element-wise unary operators (no extra parameter)
# ---------------------------------------------------------------------------

class UnaryParameterlessOperators:
    @staticmethod
    def Sign(x: pd.Series) -> pd.Series:
        return np.sign(x)

    @staticmethod
    def Sqrt(x: pd.Series) -> pd.Series:
        return np.sqrt(np.abs(x))

    @staticmethod
    def Signed_sqrt(x: pd.Series) -> pd.Series:
        return np.sign(x) * np.sqrt(np.abs(x))

    @staticmethod
    def Log1p(x: pd.Series) -> pd.Series:
        return np.log1p(x.abs())

    @staticmethod
    def sin(x: pd.Series) -> pd.Series:
        return np.sin(x.replace([np.inf, -np.inf], np.nan))

    @staticmethod
    def cos(x: pd.Series) -> pd.Series:
        return np.cos(x.replace([np.inf, -np.inf], np.nan))

    @staticmethod
    def sigmoid(x: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp((-x).clip(upper=100)))

    @staticmethod
    def softsign(x: pd.Series) -> pd.Series:
        return x / (1.0 + np.abs(x))

    @staticmethod
    def Log(x: pd.Series) -> pd.Series:
        return pd.Series(np.log(np.abs(x) + 1e-8), index=x.index, name=x.name)

    @staticmethod
    def Abs(x: pd.Series) -> pd.Series:
        return x.abs()

    @staticmethod
    def Inv(x: pd.Series) -> pd.Series:
        return 1.0 / UnaryParameterlessOperators.safe_den(x)

    @staticmethod
    def Neg(x: pd.Series) -> pd.Series:
        return -x

    @staticmethod
    def tanh(x: pd.Series) -> pd.Series:
        return pd.Series(np.tanh(x), index=x.index, name=x.name)

    @staticmethod
    def Relu(x: pd.Series) -> pd.Series:
        return pd.Series(np.maximum(0.0, x), index=x.index, name=x.name)

    @staticmethod
    def Leaky_relu(x: pd.Series) -> pd.Series:
        return pd.Series(np.where(x > 0, x, 0.01 * x), index=x.index, name=x.name)

    @staticmethod
    def Power2(x: pd.Series) -> pd.Series:
        return x ** 2

    @staticmethod
    def Power3(x: pd.Series) -> pd.Series:
        return x ** 3

    @staticmethod
    def Power15(x: pd.Series) -> pd.Series:
        return x ** 1.5

    @staticmethod
    def safe_den(y: pd.Series) -> pd.Series:
        out = y.copy()
        mask = out.abs() < 1e-8
        out.loc[mask] = np.sign(out.loc[mask]).replace(0, 1.0) * 1e-8
        return out


class UnaryParameterizedOperators:
    pass


# ---------------------------------------------------------------------------
# Element-wise binary operators (two series operands)
# ---------------------------------------------------------------------------

class BinaryOperators:
    @staticmethod
    def Add(x: pd.Series, y: pd.Series) -> pd.Series:
        return x + y

    @staticmethod
    def Subtract(x: pd.Series, y: pd.Series) -> pd.Series:
        return x - y

    @staticmethod
    def Multiply(x: pd.Series, y: pd.Series) -> pd.Series:
        return x * y

    @staticmethod
    def Divide(x: pd.Series, y: pd.Series) -> pd.Series:
        return x / UnaryParameterlessOperators.safe_den(y)

    @staticmethod
    def Min(x: pd.Series, y: pd.Series) -> pd.Series:
        return x.where(x < y, y)

    @staticmethod
    def Max(x: pd.Series, y: pd.Series) -> pd.Series:
        return x.where(x > y, y)

    @staticmethod
    def Mean(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x + y) / 2.0

    @staticmethod
    def AbsDiff(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x - y).abs()

    @staticmethod
    def RelDiff(x: pd.Series, y: pd.Series) -> pd.Series:
        return BinaryOperators.Divide(x - y, y.abs())

    @staticmethod
    def LogRatio(x: pd.Series, y: pd.Series, eps: float = 1e-8) -> pd.Series:
        return np.log((x / UnaryParameterlessOperators.safe_den(y)).abs() + eps)

    @staticmethod
    def Greater(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x > y).astype(float)

    @staticmethod
    def Less(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x < y).astype(float)

    @staticmethod
    def GreaterEqual(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x >= y).astype(float)

    @staticmethod
    def LessEqual(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x <= y).astype(float)

    @staticmethod
    def Equal(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x == y).astype(float)

    @staticmethod
    def NotEqual(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x != y).astype(float)


# ---------------------------------------------------------------------------
# Time-series binary operators (two series + window parameter)
# ---------------------------------------------------------------------------

class TimeSeriesBinaryOperators:
    @staticmethod
    def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return x.rolling(window, min_periods=1).corr(y)

    @staticmethod
    def ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return x.rolling(window, min_periods=1).cov(y)

    # @staticmethod
    # def ts_rcor(x: pd.Series, y: pd.Series, window: int = 10) -> pd.Series:
    #     return pd.Series(TimeSeriesNumbaOps.ts_rcor(x.to_numpy(), y.to_numpy(), window), index=x.index)

    @staticmethod
    def ts_beta_binary(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        cov_xy = TimeSeriesBinaryOperators.ts_cov(x, y, window)
        var_y = y.rolling(window, min_periods=1).var()
        return cov_xy / UnaryParameterlessOperators.safe_den(var_y)

    @staticmethod
    def ts_reg_resid_binary(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        roll_x = x.rolling(window, min_periods=1)
        roll_y = y.rolling(window, min_periods=1)
        mean_x = roll_x.mean()
        mean_y = roll_y.mean()
        beta = TimeSeriesBinaryOperators.ts_beta_binary(x, y, window)
        alpha = mean_x - beta * mean_y
        return x - (alpha + beta * y)

    @staticmethod
    def ts_spread_zscore(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return TimeSeriesUnaryOperators.ts_zscore(x - y, window)

    @staticmethod
    def ts_logratio_zscore(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return TimeSeriesUnaryOperators.ts_zscore(BinaryOperators.LogRatio(x, y), window)

    @staticmethod
    def ts_absdiff_mean(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return (x - y).abs().rolling(window, min_periods=1).mean()

    @staticmethod
    def ts_squared_diff_mean(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return ((x - y) ** 2).rolling(window, min_periods=1).mean()


def _collect_class_ops(cls, kind: str, arity: int) -> list[OperatorSpec]:
    specs: list[OperatorSpec] = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name)
        if not callable(attr):
            continue
        specs.append(OperatorSpec(name=name, kind=kind, arity=arity))
    return specs


class OperatorRegistry:
    """Central registry that discovers all operator group classes, builds
    ``OperatorSpec`` metadata for each method, and provides O(1) lookup
    by operator name.  Also handles token-catalog construction and
    expression normalisation for downstream evaluators."""

    def __init__(self, ts_parameters: Sequence[int]):
        self.ts_parameters = [int(value) for value in ts_parameters]
        self.operator_specs = self._build_specs()
        self.operator_names = {spec.name for spec in self.operator_specs}
        self.operator_funcs = self._build_operator_funcs()
        self.operators_by_kind = self._build_operators_by_kind()

    def _build_specs(self) -> list[OperatorSpec]:
        specs: list[OperatorSpec] = []
        specs.extend(_collect_class_ops(BinaryOperators, "binary", 2))
        specs.extend(_collect_class_ops(UnaryParameterizedOperators, "unary_parameterized", 2))
        specs.extend(_collect_class_ops(UnaryParameterlessOperators, "unary_parameterless", 1))
        specs.extend(_collect_class_ops(TimeSeriesUnaryOperators, "ts_unary", 2))
        specs.extend(_collect_class_ops(TimeSeriesBinaryOperators, "ts_binary", 3))
        return sorted(specs, key=lambda item: (item.kind, item.name))

    def _build_operator_funcs(self) -> dict[str, tuple[OperatorSpec, Any]]:
        sources = {
            "binary": BinaryOperators,
            "unary_parameterized": UnaryParameterizedOperators,
            "unary_parameterless": UnaryParameterlessOperators,
            "ts_unary": TimeSeriesUnaryOperators,
            "ts_binary": TimeSeriesBinaryOperators,
        }
        out: dict[str, tuple[OperatorSpec, Any]] = {}
        for spec in self.operator_specs:
            out[spec.name] = (spec, getattr(sources[spec.kind], spec.name))
        return out

    def _build_operators_by_kind(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spec in self.operator_specs:
            grouped[spec.kind].append(spec.as_dict())
        return dict(grouped)

    def get_operator(self, token: object) -> tuple[OperatorSpec, Any] | None:
        return self.operator_funcs.get(str(token))

    def is_operator(self, token: object) -> bool:
        return str(token) in self.operator_names

    def parse_parameter_token(self, token: object) -> int | None:
        if isinstance(token, (int, np.integer)):
            value = int(token)
            return value if value in self.ts_parameters else None
        if isinstance(token, float) and float(token).is_integer():
            value = int(token)
            return value if value in self.ts_parameters else None
        raw = str(token).strip()
        if raw.startswith("TS_"):
            raw = raw[3:]
        if raw.startswith("-") or not raw.isdigit():
            return None
        value = int(raw)
        return value if value in self.ts_parameters else None

    def parameter_token(self, value: int) -> str:
        return f"TS_{int(value)}"

    def build_token_catalog(self, feature_names: Sequence[str]) -> dict[str, Any]:
        feature_tokens = list(feature_names)
        ts_param_tokens = [self.parameter_token(value) for value in self.ts_parameters]
        operator_tokens = [spec.name for spec in self.operator_specs]
        vocab = feature_tokens + ts_param_tokens + operator_tokens + ["<EOS>"]
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        id_to_token = {str(idx): token for idx, token in enumerate(vocab)}
        return {
            "feature_tokens": feature_tokens,
            "ts_param_tokens": ts_param_tokens,
            "operator_tokens": operator_tokens,
            "vocab": vocab,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "ts_param_start": len(feature_tokens),
            "op_start": len(feature_tokens) + len(ts_param_tokens),
            "eos_token_id": len(vocab) - 1,
        }

    def build_capability(self, feature_names: Sequence[str], data_path: str | None = None) -> dict[str, Any]:
        feature_names = list(feature_names)
        token_catalog = self.build_token_catalog(feature_names)
        return {
            "data_path": data_path,
            "feature_tokens": feature_names,
            "ts_parameters": list(self.ts_parameters),
            "operator_specs": [spec.as_dict() for spec in self.operator_specs],
            "operators_by_kind": self.operators_by_kind,
            "token_catalog": token_catalog,
        }

