"""Persistence helpers for AlphaPool runtime artifacts."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .feature_snapshot import (
    DUMPED_FEATURES_SUBDIR,
    sync_distributions,
    write_dumped_feature,
)


def write_persist_manifest(log_dir: str, *, source: str = "config") -> str:
    """Write the manifest that documents AlphaPool runtime outputs."""
    root = os.path.abspath(os.path.expanduser(log_dir))
    os.makedirs(root, exist_ok=True)
    manifest: Dict[str, Any] = {
        "version": 2,
        "source": source,
        "root": root,
        "description": (
            "AlphaPool runtime artifacts. "
            "current_pool/ and best_pool/ are self-contained loadable pool directories "
            "(pool_state.json + factors/ + distributions/). "
            "dumped_features/ is a permanent per-feature archive (one sub-dir per factor_id)."
        ),
        "relative_paths": [
            "pool_history.jsonl",
            "actor_meta.json",
            "alphapool_persist_manifest.json",
            "current_pool/",
            "best_pool/",
            "dumped_features/",
        ],
    }
    path = os.path.join(root, "alphapool_persist_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def append_pool_history_jsonl(log_dir: str, record: Dict[str, Any]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "pool_history.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_actor_meta(
    path: str,
    *,
    step: int,
    best_pool_score: Optional[float],
    best_pool_step: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "step": int(step),
        "best_pool_score": best_pool_score,
        "best_pool_step": best_pool_step,
    }
    if extra:
        payload.update(extra)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_actor_meta(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_current_pool_snapshot(
    log_dir: str,
    pool_state,
    entries: List,
) -> Optional[str]:
    """Save current pool to log_dir/current_pool/.

    Writes pool_state.json + factors/ + distributions/, removing any stale
    files for entries no longer in the pool.
    """
    if not entries:
        return None
    pool_dir = os.path.join(log_dir, "current_pool")
    pool_state.save(pool_dir)
    sync_distributions(pool_dir, entries)
    return pool_dir


def write_best_pool_snapshot(
    log_dir: str,
    pool_state,
    entries: List,
    step: int,
    score: float,
) -> Optional[str]:
    """Save best pool to log_dir/best_pool/.

    Writes pool_state.json + factors/ + distributions/ + best_meta.json,
    removing any stale files for entries no longer in the pool.
    """
    if not entries:
        return None
    pool_dir = os.path.join(log_dir, "best_pool")
    pool_state.save(pool_dir)
    sync_distributions(pool_dir, entries)
    meta_path = os.path.join(pool_dir, "best_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"step": int(step), "score": float(score)},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return pool_dir


def write_dumped_features(
    log_dir: str,
    entries: List,
    importance_arr: Optional[np.ndarray] = None,
    *,
    step: Optional[int] = None,
    feature_metrics_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Write / update permanent per-feature archives under log_dir/dumped_features/.

    Each active pool entry gets its own sub-directory keyed by factor_id.
    The archive is append-only: features that leave the pool are NOT removed
    from dumped_features/ – they remain as historical records.

    Args:
        log_dir: AlphaPool log directory (parent of current_pool/, best_pool/, …).
        entries: Current pool entries (aligned with *importance_arr*).
        importance_arr: Importance weights, one per entry in *entries* order.
        step: Current pool step counter (written to info.json).
        feature_metrics_map: Optional mapping from feature name to its
            evaluation-time feature_metrics dict.
    """
    dump_dir = os.path.join(log_dir, DUMPED_FEATURES_SUBDIR)
    os.makedirs(dump_dir, exist_ok=True)

    for i, entry in enumerate(entries):
        importance: Optional[float] = None
        if importance_arr is not None and i < len(importance_arr):
            importance = float(importance_arr[i])
        feat_metrics: Optional[Dict[str, Any]] = None
        if feature_metrics_map is not None:
            feat_metrics = feature_metrics_map.get(entry.name)
        write_dumped_feature(
            dump_dir,
            entry,
            importance=importance,
            feature_metrics=feat_metrics,
            step=step,
        )

    return dump_dir
