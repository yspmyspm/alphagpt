"""
基于 myops 的算子注册中心：
- 返回统一 op 元信息，供模型词表、约束解码、VM 执行共用。
"""
from config import ModelConfig
from helpers.myops import binary, ts_binary, ts_unary, unary_parameterized, unary_parameterless


def _collect_class_ops(cls, kind, arity):
    out = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name)
        if not callable(attr):
            continue
        out.append((name, attr, kind, arity))
    return out


def get_op_specs():
    """
    返回 [(name, func, kind, arity), ...]
    kind:
      - binary: 2 个 Feature
      - unary_parameterless: 1 个 Feature
      - unary_parameterized: 1 个 Feature + 1 个 Parameter
      - ts_unary: 1 个 Feature + 1 个 Parameter
      - ts_binary: 2 个 Feature + 1 个 Parameter
    arity 为栈弹出元素数量（含参数）。
    """
    specs = []
    specs.extend(_collect_class_ops(binary, "binary", 2))
    specs.extend(_collect_class_ops(unary_parameterized, "unary_parameterized", 2))
    specs.extend(_collect_class_ops(unary_parameterless, "unary_parameterless", 1))
    specs.extend(_collect_class_ops(ts_unary, "ts_unary", 2))
    specs.extend(_collect_class_ops(ts_binary, "ts_binary", 3))
    return specs


def get_ops():
    """兼容旧接口：返回 [(name, func, arity), ...]。"""
    return [(name, func, arity) for name, func, _, arity in get_op_specs()]


def get_ts_parameters():
    return [int(v) for v in getattr(ModelConfig, "TS_PARAMETERS", [1, 5, 10, 30, 60, 120, 1440])]
