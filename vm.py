import numpy as np
import pandas as pd

from helpers.ops import get_op_specs, get_ts_parameters
from factors import FeatureEngineer


class StackVM:
    """基于 pd.Series 的栈式 VM，算子从 SeriesOps 类遍历获取。"""

    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.ts_parameters = get_ts_parameters()
        self.param_start = self.feat_offset
        self.param_end = self.param_start + len(self.ts_parameters)

        op_specs = get_op_specs()
        self.op_start = self.param_end
        self.op_map = {
            i + self.op_start: (name, func, kind, arity)
            for i, (name, func, kind, arity) in enumerate(op_specs)
        }
        self.arity_map = {tid: arity for tid, (_, _, _, arity) in self.op_map.items()}

    def execute(self, formula_tokens, features: list[pd.Series]):
        """
        features: List[pd.Series]，按 token id 顺序，均带时间索引。
        返回 pd.Series 或 None，保留时间索引。
        """
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    stack.append(features[token].copy())
                elif self.param_start <= token < self.param_end:
                    stack.append(self.ts_parameters[token - self.param_start])
                elif token in self.op_map:
                    name, func, kind, arity = self.op_map[token]

                    if kind == "binary":
                        if len(stack) < 2:
                            return None
                        y = stack.pop()
                        x = stack.pop()
                        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
                            return None
                        res = func(x, y)
                    elif kind in ("unary_parameterized", "ts_unary"):
                        if len(stack) < 2:
                            return None
                        param = stack.pop()
                        x = stack.pop()
                        if not isinstance(x, pd.Series) or not isinstance(param, (int, float)):
                            return None
                        res = func(x, int(param))
                    elif kind == "unary_parameterless":
                        if len(stack) < 1:
                            return None
                        x = stack.pop()
                        if not isinstance(x, pd.Series):
                            return None
                        res = func(x)
                    elif kind == "ts_binary":
                        if len(stack) < 3:
                            return None
                        param = stack.pop()
                        y = stack.pop()
                        x = stack.pop()
                        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series) or not isinstance(param, (int, float)):
                            return None
                        res = func(x, y, int(param))
                    else:
                        return None

                    if res is None:
                        return None
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            return None
        except Exception:
            return None
