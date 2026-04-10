"""Alpha pool state persistence and normalized entry helpers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

STATE_MANIFEST = "pool_state.json"
FACTORS_SUBDIR = "factors"
STATE_MANIFEST_VERSION = 2


def copy_importance(importance: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if importance is None:
        return None
    return np.asarray(importance, dtype=np.float64).ravel().copy()


def copy_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return dict(metrics) if metrics is not None else None


def _maybe_parse_datetime_index(series: pd.Series) -> pd.Series:
    out = series.copy()
    try:
        parsed = pd.to_datetime(out.index)
    except Exception:
        return out
    if getattr(parsed, "isna", None) is not None and bool(parsed.isna().any()):
        return out
    out.index = parsed
    return out


def factor_dense_values(factor: pd.Series) -> np.ndarray:
    return factor.to_numpy(dtype=np.float64, copy=True)


def compute_factor_id(name: str) -> str:
    """Deterministic short hash of the expression name, used as file-safe identifier."""
    return hashlib.blake2b(str(name).encode("utf-8"), digest_size=12).hexdigest()


def factor_file_name(factor_id: str) -> str:
    return f"{factor_id}.feather"


@dataclass
class PoolEntry:
    """One pool entry with cached metadata for repeated hot-path access."""

    name: str
    factor: pd.Series
    factor_id: Optional[str] = None
    dense_values: Optional[np.ndarray] = None

    def normalized(self) -> "PoolEntry":
        factor = self.factor.astype(float, copy=False)
        factor_id = str(self.factor_id) if self.factor_id else compute_factor_id(self.name)
        dense_values = self.dense_values
        if dense_values is None:
            dense_values = factor_dense_values(factor)
        else:
            dense_values = np.asarray(dense_values, dtype=np.float64)
        return PoolEntry(
            name=str(self.name),
            factor=factor,
            factor_id=factor_id,
            dense_values=dense_values,
        )

    def copy(self) -> "PoolEntry":
        normalized = self.normalized()
        return PoolEntry(
            name=normalized.name,
            factor=normalized.factor.copy(),
            factor_id=normalized.factor_id,
            dense_values=normalized.dense_values.copy(),
        )

    def to_snapshot_payload(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "name": normalized.name,
            "factor": normalized.factor.copy(),
            "factor_id": normalized.factor_id,
            "dense_values": normalized.dense_values.copy(),
        }

    @classmethod
    def from_snapshot_payload(cls, payload: Dict[str, Any]) -> "PoolEntry":
        factor = payload.get("factor")
        if not isinstance(factor, pd.Series):
            raise ValueError("snapshot payload.factor must be a pandas Series")
        return cls(
            name=str(payload.get("name", "")),
            factor=factor,
            factor_id=payload.get("factor_id"),
            dense_values=payload.get("dense_values"),
        ).normalized()


class AlphaPoolState:
    """In-memory pool state with stable-id persistence helpers and fit cache."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.entries: List[PoolEntry] = []
        self._fit_cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def _normalize_entry(entry: PoolEntry) -> PoolEntry:
        if not isinstance(entry, PoolEntry):
            raise TypeError(f"expected PoolEntry, got {type(entry).__name__}")
        return entry.normalized()

    def replace_entries(self, new_entries: List[PoolEntry]) -> None:
        self.entries = [self._normalize_entry(entry) for entry in new_entries[: self.capacity]]
        self.clear_fit_cache()

    def export_entries_snapshot(self) -> List[Dict[str, Any]]:
        return [entry.to_snapshot_payload() for entry in self.entries]

    def import_entries_snapshot(self, snapshot: List[Dict[str, Any]]) -> None:
        self.replace_entries([PoolEntry.from_snapshot_payload(item) for item in snapshot])

    # ------------------------------------------------------------------
    # Fit cache
    # ------------------------------------------------------------------

    def entry_factor_ids(self, entries: Optional[List[PoolEntry]] = None) -> List[str]:
        pool_entries = self.entries if entries is None else list(entries)
        return [str(entry.normalized().factor_id) for entry in pool_entries]

    def clear_fit_cache(self) -> None:
        self._fit_cache = None

    def set_fit_cache(
        self,
        entries: List[PoolEntry],
        *,
        importance: Optional[np.ndarray],
        metrics: Optional[Dict[str, Any]],
    ) -> None:
        copied_importance = copy_importance(importance)
        factor_ids = self.entry_factor_ids(entries)
        if copied_importance is None:
            copied_importance = np.zeros(len(factor_ids), dtype=np.float64)
        if copied_importance.shape[0] != len(factor_ids):
            self.clear_fit_cache()
            return
        self._fit_cache = {
            "entry_factor_ids": factor_ids,
            "importance": copied_importance,
            "metrics": copy_metrics(metrics),
        }

    def get_fit_cache(
        self, entries: Optional[List[PoolEntry]] = None,
    ) -> Optional[Dict[str, Any]]:
        cache = self._fit_cache
        if cache is None:
            return None
        factor_ids = self.entry_factor_ids(entries)
        cached_ids = list(cache.get("entry_factor_ids") or [])
        cached_importance = copy_importance(cache.get("importance"))
        if cached_ids != factor_ids or cached_importance is None:
            return None
        if cached_importance.shape[0] != len(factor_ids):
            return None
        return {
            "entry_factor_ids": cached_ids,
            "importance": cached_importance,
            "metrics": copy_metrics(cache.get("metrics")),
        }

    def export_fit_cache(self) -> Optional[Dict[str, Any]]:
        cache = self.get_fit_cache()
        if cache is None:
            return None
        return {
            "entry_factor_ids": list(cache["entry_factor_ids"]),
            "importance": cache["importance"].tolist(),
            "metrics": copy_metrics(cache.get("metrics")),
        }

    def restore_fit_cache(self, payload: Any) -> None:
        self.clear_fit_cache()
        if not isinstance(payload, dict):
            return
        factor_ids = [str(item) for item in payload.get("entry_factor_ids") or []]
        current_ids = self.entry_factor_ids()
        importance = copy_importance(payload.get("importance"))
        if factor_ids != current_ids or importance is None or importance.shape[0] != len(current_ids):
            return
        metrics = payload.get("metrics")
        self._fit_cache = {
            "entry_factor_ids": factor_ids,
            "importance": importance,
            "metrics": copy_metrics(metrics) if isinstance(metrics, dict) else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Write ``directory/pool_state.json`` plus stable ``directory/factors/*.feather`` files.
        Existing factor files are reused when the same factor_id appears again.
        Stale feather files for entries no longer in the pool are removed.
        """
        os.makedirs(directory, exist_ok=True)
        fac_dir = os.path.join(directory, FACTORS_SUBDIR)
        os.makedirs(fac_dir, exist_ok=True)

        normalized_entries = [self._normalize_entry(e) for e in self.entries]
        current_fids = {e.factor_id for e in normalized_entries}

        # Build importance map from fit cache (factor_id -> importance value).
        importance_map: Dict[str, float] = {}
        fit_cache = self.get_fit_cache()
        if fit_cache is not None:
            cached_ids = fit_cache.get("entry_factor_ids") or []
            cached_imp = fit_cache.get("importance")
            if cached_imp is not None:
                for fid, imp in zip(cached_ids, cached_imp):
                    importance_map[str(fid)] = float(imp)

        # Remove feather files for entries that are no longer in the pool.
        for fname in os.listdir(fac_dir):
            if not fname.endswith(".feather"):
                continue
            fid = fname[: -len(".feather")]
            if fid not in current_fids:
                try:
                    os.remove(os.path.join(fac_dir, fname))
                except OSError:
                    pass

        manifest: Dict[str, Any] = {
            "version": STATE_MANIFEST_VERSION,
            "capacity": int(self.capacity),
            "entries": [],
        }
        for normalized in normalized_entries:
            rel = os.path.join(FACTORS_SUBDIR, factor_file_name(normalized.factor_id))
            fp = os.path.join(directory, rel)
            if not os.path.isfile(fp):
                df = pd.DataFrame({"factor": normalized.factor.values}, index=normalized.factor.index)
                df.index.name = df.index.name or "timestamp"
                df.reset_index().to_feather(fp)

            fv = normalized.dense_values
            fv_finite = fv[np.isfinite(fv)]
            entry_info: Dict[str, Any] = {
                "name": normalized.name,
                "factor_id": normalized.factor_id,
                "factor_file": rel,
            }
            if normalized.factor_id in importance_map:
                entry_info["importance"] = importance_map[normalized.factor_id]
            if len(fv_finite) >= 4:
                s = pd.Series(fv_finite)
                entry_info["skew"] = float(s.skew())
                entry_info["kurt"] = float(s.kurt())
            manifest["entries"].append(entry_info)

        mp = os.path.join(directory, STATE_MANIFEST)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str, capacity: int) -> "AlphaPoolState":
        mp = os.path.join(directory, STATE_MANIFEST)
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"missing {STATE_MANIFEST}: {directory}")
        with open(mp, encoding="utf-8") as f:
            manifest = json.load(f)
        p = cls(capacity)
        cap = int(manifest.get("capacity", capacity))
        p.capacity = min(cap, capacity)
        entries: List[PoolEntry] = []
        for item in manifest.get("entries", []):
            rel = item["factor_file"]
            fp = os.path.join(directory, rel)
            df = pd.read_feather(fp)
            idx_col = df.columns[0]
            factor = pd.Series(df["factor"].values, index=df[idx_col])
            factor = _maybe_parse_datetime_index(factor)
            entries.append(
                PoolEntry(
                    name=str(item["name"]),
                    factor=factor,
                    factor_id=item.get("factor_id"),
                ).normalized()
            )
        p.entries = entries[: p.capacity]
        return p

