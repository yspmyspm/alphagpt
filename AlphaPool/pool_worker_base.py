"""Shared base class for AlphaPool coordinator and worker actors."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pool_state import copy_importance, copy_metrics


class AlphaPoolActorBase:
	"""Mixin with scoring / metrics helpers shared by coordinator and workers.

	Subclasses must set the following attributes before calling any of
	these methods:

	- ``self._returns``  (pd.Series)
	- ``self._data_idx`` (pd.Index)
	- ``self._missing_threshold`` (float)
	- ``self.trainer``   (ensemble trainer with fit_and_predict / diversity_score)
	- ``self.scorer``    (scorer with score_series)
	"""

	# ------------------------------------------------------------------
	# Series validation
	# ------------------------------------------------------------------

	def _validate_and_align_series(self, series: pd.Series, name: str) -> pd.Series:
		"""Align one factor to the pool index and enforce minimum coverage."""
		idx = self._data_idx
		valid_start, valid_end = idx.min(), idx.max()
		series_start, series_end = series.index.min(), series.index.max()
		if series_end < valid_start or series_start > valid_end:
			raise ValueError(
				f"Series '{name}' range [{series_start}, {series_end}] "
				f"does not overlap with pool range [{valid_start}, {valid_end}]"
			)
		aligned = series.reindex(idx)
		coverage = float(aligned.notna().mean())
		min_coverage = 1.0 - self._missing_threshold
		if coverage < min_coverage:
			raise ValueError(
				f"Series '{name}' coverage {coverage:.1%} below minimum {min_coverage:.1%}"
			)
		return aligned

	# ------------------------------------------------------------------
	# Scoring / metrics helpers
	# ------------------------------------------------------------------

	def _compute_metrics_for_entries(
		self,
		entries: List[Any],
	) -> Tuple[Optional[Dict[str, Any]], np.ndarray]:
		"""Fit the ensemble and score the resulting combo prediction."""
		if not entries:
			return None, np.zeros(0, dtype=np.float64)

		fit = self.trainer.fit_and_predict(
			entries, self._returns, self._data_idx, scorer=self.scorer,
		)
		_ = copy_importance(fit.importance)
		if _ is None:
			return None, np.zeros(0, dtype=np.float64)
		else:
			importance = _
		pred = fit.prediction

		result: Dict[str, Any] = {"score": None}
		if len(pred) == 0:
			return result, importance

		score_result = self._score_series(pred)
		result = self._metrics_from_score_result(score_result)
		return result, importance

	def _score_series(self, series: pd.Series):
		return self.scorer.score_series(
			series, self._returns, subset_index=self._data_idx,
		)

	def _metrics_from_score_result(
		self, score_result, *, score_hint: Optional[float] = None,
	) -> Dict[str, Any]:
		score_value = None if score_hint is None else float(score_hint)
		if score_value is None and getattr(score_result, "score", None) is not None:
			score_value = float(score_result.score)
		result: Dict[str, Any] = {"score": score_value}
		metrics = dict(getattr(score_result, "metrics", {}) or {})
		metrics.pop("score", None)
		metrics.pop("weighted_score", None)
		result.update(metrics)
		return result

	def _compute_factor_metrics(self, factor: pd.Series) -> Dict[str, Any]:
		score_result = self._score_series(factor)
		metrics = self._metrics_from_score_result(score_result)
		metrics.pop("score", None)
		return metrics

	# ------------------------------------------------------------------
	# Static helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _expr_eval_failure(name: str, reason: str) -> Dict[str, Any]:
		return {
			"name": str(name),
			"valid": False,
			"invalid_reason": str(reason),
			"feature_metrics": {},
			"diversity_metrics": {},
			"pool_metrics_before": None,
			"pool_metrics_if_added": None,
			"gain_metrics": {},
		}
