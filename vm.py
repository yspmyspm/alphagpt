import numpy as np
import pandas as pd

from helpers.ops import get_ops
from factors import FeatureEngineer


class StackVM:
    """基于 pd.Series 的栈式 VM，算子从 SeriesOps 类遍历获取。"""

    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        ops = get_ops()
        self.op_map = {i + self.feat_offset: (name, func, arity) for i, (name, func, arity) in enumerate(ops)}
        self.arity_map = {i + self.feat_offset: arity for i, (_, _, arity) in enumerate(ops)}

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
                elif token in self.op_map:
                    name, func, arity = self.op_map[token]
                    if len(stack) < arity:
                        return None
                    args = [stack.pop() for _ in range(arity)]
                    args.reverse()
                    res = func(*args)
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
