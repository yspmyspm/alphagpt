"""Dynamic component loading helpers."""
from __future__ import annotations

import importlib
from typing import Any


def load_symbol(class_path: str) -> Any:
    """
    Load ``module.submodule:Symbol`` dynamically.
    """
    module_path, sep, symbol_name = str(class_path).strip().partition(":")
    if not sep or not module_path or not symbol_name:
        raise ValueError(
            f"Invalid class_path {class_path!r}; expected 'package.module:SymbolName'"
        )
    module = importlib.import_module(module_path)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(
            f"Symbol {symbol_name!r} not found in module {module_path!r}"
        ) from exc
