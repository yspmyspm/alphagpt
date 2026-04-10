"""Read-only evaluation worker for the AlphaPool system."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ALPHAPOOL_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ALPHAPOOL_ROOT not in sys.path:
	sys.path.insert(0, _ALPHAPOOL_ROOT)

from utils.component_loader import load_symbol
from pool_state import AlphaPoolState, PoolEntry, copy_importance, copy_metrics
from pool_worker_base import AlphaPoolActorBase
from utils.metrics import compute_compliance_only, missing_ratio_on_common


class AlphaPoolWorkerActor(AlphaPoolActorBase):
	"""Read-only evaluation worker for candidate alpha factors.

	Each worker holds a lazily-synchronised snapshot of the pool and
	evaluates individual candidate factors against that snapshot.
	Workers never mutate the authoritative pool.
	"""

	def __init__(
		self,
		*,
		returns: pd.Series,
		data_idx: pd.Index,
		pool_capacity: int,
		trainer_spec: Optional[Dict[str, Any]] = None,
		scorer_spec: Optional[Dict[str, Any]] = None,
		feature_filter_params: Optional[Dict[str, Any]] = None,
		quality_gate_mode: str = "smooth",
		quality_gate_params: Optional[Dict[str, Any]] = None,
		**_ignored,
	):
		self._returns = returns
		self._data_idx = data_idx
		self._feature_filter_params = dict(feature_filter_params or {})
		self._missing_threshold = float(self._feature_filter_params.get("missing_threshold", 0.3))
		self._low_std_threshold = float(self._feature_filter_params.get("low_std_threshold", 1e-4))

		self._quality_gate_mode = str(quality_gate_mode or "smooth").strip().lower()
		if self._quality_gate_mode not in ("smooth", "strict"):
			raise ValueError(
				f"quality_gate_mode must be 'smooth' or 'strict', got {quality_gate_mode!r}"
			)
		self._quality_gate_params = dict(quality_gate_params or {})

		self._cached_snapshot_version = -1
		self.pool = AlphaPoolState(capacity=pool_capacity)

		trainer_cfg = dict(trainer_spec or {})
		trainer_cls = load_symbol(trainer_cfg["class_path"])
		self.trainer = trainer_cls(**dict(trainer_cfg.get("params") or {}))

		scorer_cfg = dict(scorer_spec or {})
		scorer_cls = load_symbol(scorer_cfg["class_path"])
		self.scorer = scorer_cls(**dict(scorer_cfg.get("params") or {}))

		self._pool_metrics: Optional[Dict[str, Any]] = None
		self._pool_importance: Optional[np.ndarray] = None

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def evaluate_factor(self, name: str, series: pd.Series) -> Dict[str, Any]:
		"""Evaluate a single candidate factor against the current pool snapshot."""
		pool_entries = list(self.pool.entries)
		result = self._evaluate_one(
			name, series, pool_entries,
			pool_metrics_before=copy_metrics(self._pool_metrics),
		)
		return result

	def apply_pool_delta(
		self, version: int, delta: Dict[str, Any],
	) -> Dict[str, Any]:
		"""Apply an incremental pool update from the coordinator."""
		if int(version) == self._cached_snapshot_version:
			return {
				"ok": True,
				"changed": False,
				"version": int(version),
				"pool_size": len(self.pool.entries),
			}

		left_ids = set(delta.get("left_ids") or [])
		joined = delta.get("joined") or []
		entry_manifest = delta.get("entry_manifest") or []

		entry_map = {e.normalized().factor_id: e for e in self.pool.entries}

		for fid in left_ids:
			entry_map.pop(fid, None)

		for payload in joined:
			entry = PoolEntry.from_snapshot_payload(payload)
			entry_map[entry.factor_id] = entry

		new_entries = []
		for item in entry_manifest:
			fid = item["factor_id"]
			if fid in entry_map:
				new_entries.append(entry_map[fid])

		self.pool.replace_entries(new_entries)
		self._cached_snapshot_version = int(version)
		self._pool_metrics = copy_metrics(delta.get("pool_metrics"))
		self._pool_importance = copy_importance(delta.get("importance"))
		return {
			"ok": True,
			"changed": True,
			"version": int(version),
			"pool_size": len(self.pool.entries),
		}

	# ------------------------------------------------------------------
	# Evaluation core
	# ------------------------------------------------------------------

	def _evaluate_one(
		self,
		name: str,
		series: pd.Series,
		pool_entries: List[PoolEntry],
		*,
		pool_metrics_before: Optional[Dict[str, Any]],
	) -> Dict[str, Any]:
		"""Evaluate one candidate against the current pool snapshot."""

		# Stage 1: Alignment, missing/std check, compliance
		try:
			aligned = self._validate_and_align_series(series, name)
		except ValueError as exc:
			return self._expr_eval_failure(name, str(exc))

		candidate = PoolEntry(name=str(name), factor=aligned).normalized()

		missing_ratio = missing_ratio_on_common(aligned, self._returns)
		if missing_ratio > self._missing_threshold:
			return {
				**self._expr_eval_failure(name, "missing_high"),
				"feature_metrics": {"missing_ratio": float(missing_ratio)},
			}

		std_val = aligned.std()
		if std_val < self._low_std_threshold:
			return {
				**self._expr_eval_failure(name, "low_std"),
				"feature_metrics": {
					"missing_ratio": float(missing_ratio),
					"std": float(std_val),
				},
			}

		comp = compute_compliance_only(
			aligned,
			self._returns,
			use_smooth_reward=(self._quality_gate_mode == "smooth"),
			**self._quality_gate_params,
		)
		if comp is None:
			return {
				**self._expr_eval_failure(name, "compliance_fail"),
				"feature_metrics": {
					"missing_ratio": float(missing_ratio),
					"std": float(std_val),
				},
			}
		s_finite, s_dist, s_half, compliance = comp

		feature_metrics = self._compute_factor_metrics(aligned)
		feature_metrics["missing_ratio"] = float(missing_ratio)
		feature_metrics["std"] = float(std_val)
		feature_metrics["compliance"] = float(compliance)
		feature_metrics["compliance_components"] = {
			"finite_ratio": float(s_finite),
			"distribution": float(s_dist),
			"halflife": float(s_half),
		}
		feature_metrics["skew"] = float(aligned.skew())
		feature_metrics["kurt"] = float(aligned.kurt())

		# Stage 2: Diversity
		max_corr = self.trainer.diversity_score(aligned, pool_entries, self._data_idx)

		# Stage 3: Hypothetical pool metrics
		pool_metrics_if_added, importance_arr = self._compute_metrics_for_entries(
			pool_entries + [candidate],
		)
		if pool_metrics_if_added is None or pool_metrics_if_added.get("score") is None:
			return {
				**self._expr_eval_failure(name, "fit_failed"),
				"feature_metrics": feature_metrics,
			}

		# Candidate importance in the hypothetical pool (last entry's weight)
		candidate_importance = (
			float(importance_arr[-1])
			if importance_arr is not None and len(importance_arr) > 0
			else None
		)

		# Quantile info: [0%, 10%, ..., 90%, 100%] of aligned factor values
		_q_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		feature_metrics["quantiles"] = [
			round(float(v), 8) for v in aligned.quantile(_q_levels).tolist()
		]

		# Stage 4: Assemble result
		return {
			"name": str(name),
			"valid": True,
			"invalid_reason": None,
			"feature_metrics": feature_metrics,
			"diversity_metrics": {
				"max_corr": float(max_corr),
				"diversity_multiplier": float(1.0 - max_corr),
			},
			"candidate_importance": candidate_importance,
			"pool_metrics_before": self._copy_external_pool_metrics(pool_metrics_before),
			"pool_metrics_if_added": self._copy_external_pool_metrics(pool_metrics_if_added),
			"gain_metrics": self._gain_metrics_from_pool_metrics(
				pool_metrics_before,
				pool_metrics_if_added,
				pool_size_before=len(pool_entries),
			),
		}

	def _gain_metrics_from_pool_metrics(
		self,
		before: Optional[Dict[str, Any]],
		after: Optional[Dict[str, Any]],
		*,
		pool_size_before: int,
	) -> Dict[str, Any]:
		out: Dict[str, Any] = {
			"pool_size_before": int(pool_size_before),
			"pool_size_if_added": int(pool_size_before + 1),
		}
		for key in (
			"overall_ic", "daily_ic_mean", "monthly_ic_mean",
			"daily_ic_std", "monthly_ic_std",
			"daily_icir_raw", "monthly_icir_raw",
			"daily_icir", "monthly_icir",
			"daily_missing_penalty", "monthly_missing_penalty",
			"ic_score", "daily_coverage", "monthly_coverage",
		):
			raw_after = (after or {}).get(key)
			after_value = None if raw_after is None else float(raw_after)
			raw_before = (before or {}).get(key)
			before_value = None if raw_before is None else float(raw_before)
			if after_value is not None:
				out[f"{key}_if_added"] = after_value
				out[f"{key}_delta"] = after_value - (before_value or 0.0)
		return out

	# ------------------------------------------------------------------
	# Static helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _copy_external_pool_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		if metrics is None:
			return None
		out = dict(metrics)
		out.pop("score", None)
		out.pop("weighted_score", None)
		return out
