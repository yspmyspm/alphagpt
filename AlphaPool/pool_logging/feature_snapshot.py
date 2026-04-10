"""
Generate / sync distribution plots for pool entries and maintain dumped_features/.

Pool directories (current_pool/, best_pool/) must already contain pool_state.json +
factors/ (written by AlphaPoolState.save). This module adds or syncs
distributions/*.png on top, and maintains a permanent dumped_features/ archive.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pool_state import PoolEntry
from .distribution_plot import plot_factor_distribution

DISTRIBUTIONS_SUBDIR = "distributions"
DUMPED_FEATURES_SUBDIR = "dumped_features"


def sync_distributions(
    pool_dir: str,
    entries: List[PoolEntry],
) -> None:
    """Sync pool_dir/distributions/ to match exactly the given entries.

    - PNG files for factor_ids no longer in *entries* are removed.
    - Missing PNGs for current entries are generated.
    """
    dist_dir = os.path.join(pool_dir, DISTRIBUTIONS_SUBDIR)
    os.makedirs(dist_dir, exist_ok=True)

    current_fids = {entry.normalized().factor_id for entry in entries}

    # Remove stale PNG files.
    for fname in os.listdir(dist_dir):
        if not fname.endswith(".png"):
            continue
        stem = fname[:-4]  # strip .png
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[0] not in current_fids:
            try:
                os.remove(os.path.join(dist_dir, fname))
            except OSError:
                pass

    # Generate missing PNGs.
    for entry in entries:
        normalized = entry.normalized()
        fid = normalized.factor_id
        raw_path = os.path.join(dist_dir, f"{fid}_raw.png")
        clipped_path = os.path.join(dist_dir, f"{fid}_clipped.png")
        if os.path.isfile(raw_path) and os.path.isfile(clipped_path):
            continue
        fv = normalized.dense_values
        fv = fv[np.isfinite(fv)]
        plot_factor_distribution(fv, raw_path, clipped_path, normalized.name)


def write_dumped_feature(
    dump_dir: str,
    entry: PoolEntry,
    *,
    importance: Optional[float] = None,
    feature_metrics: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
) -> str:
    """Write or update one feature's permanent archive under dump_dir/{factor_id}/.

    Layout::

        dump_dir/{factor_id}/
            factor.feather          – raw factor values (written once)
            distribution_raw.png    – full distribution (written once)
            distribution_clipped.png – q01–q99 distribution (written once)
            info.json               – name, skew, kurt, importance, metrics, steps
    """
    normalized = entry.normalized()
    fid = normalized.factor_id
    feat_dir = os.path.join(dump_dir, fid)
    os.makedirs(feat_dir, exist_ok=True)

    # Factor values – written once (stable by factor_id).
    fac_path = os.path.join(feat_dir, "factor.feather")
    if not os.path.isfile(fac_path):
        df = pd.DataFrame(
            {"factor": normalized.factor.values},
            index=normalized.factor.index,
        )
        df.index.name = df.index.name or "timestamp"
        df.reset_index().to_feather(fac_path)

    # Distribution plots – written once.
    raw_path = os.path.join(feat_dir, "distribution_raw.png")
    clipped_path = os.path.join(feat_dir, "distribution_clipped.png")
    if not os.path.isfile(raw_path) or not os.path.isfile(clipped_path):
        fv = normalized.dense_values
        fv = fv[np.isfinite(fv)]
        plot_factor_distribution(fv, raw_path, clipped_path, normalized.name)

    # Info JSON – always refreshed so importance / steps stay current.
    info_path = os.path.join(feat_dir, "info.json")
    info: Dict[str, Any] = {}
    if os.path.isfile(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = {}

    info["name"] = normalized.name
    info["factor_id"] = fid

    fv_finite = normalized.dense_values
    fv_finite = fv_finite[np.isfinite(fv_finite)]
    if len(fv_finite) >= 4:
        s = pd.Series(fv_finite)
        info["skew"] = float(s.skew())
        info["kurt"] = float(s.kurt())

    if importance is not None:
        info["importance"] = float(importance)

    if feature_metrics is not None:
        info["feature_metrics"] = dict(feature_metrics)

    if step is not None:
        if "first_seen_step" not in info:
            info["first_seen_step"] = int(step)
        info["last_seen_step"] = int(step)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return feat_dir
