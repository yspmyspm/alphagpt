"""AlphaPool coordinator: owns the authoritative pool state and dispatches
evaluation to read-only workers."""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ray

_ALPHAPOOL_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ALPHAPOOL_ROOT not in sys.path:
	sys.path.insert(0, _ALPHAPOOL_ROOT)

from utils.component_loader import load_symbol
from worker import AlphaPoolWorkerActor
from pool_logging.actor import (
	append_pool_history_jsonl,
	load_actor_meta,
	write_actor_meta,
	write_best_pool_snapshot,
	write_current_pool_snapshot,
	write_dumped_features,
	write_persist_manifest,
)
from pool_state import STATE_MANIFEST, AlphaPoolState, PoolEntry, copy_importance, copy_metrics
from pool_worker_base import AlphaPoolActorBase
from utils.metrics import compute_compliance_only, missing_ratio_on_common


class _AsyncWriterPrefRWLock:
	"""Writer-preferred async read-write lock."""

	def __init__(self) -> None:
		self._cond = asyncio.Condition()
		self._active_readers = 0
		self._writer_active = False
		self._waiting_writers = 0

	async def acquire_read(self) -> None:
		async with self._cond:
			while self._writer_active or self._waiting_writers > 0:
				await self._cond.wait()
			self._active_readers += 1

	async def release_read(self) -> None:
		async with self._cond:
			self._active_readers -= 1
			if self._active_readers < 0:
				self._active_readers = 0
			self._cond.notify_all()

	async def acquire_write(self) -> None:
		async with self._cond:
			self._waiting_writers += 1
			try:
				while self._writer_active or self._active_readers > 0:
					await self._cond.wait()
				self._writer_active = True
			finally:
				self._waiting_writers -= 1

	async def release_write(self) -> None:
		async with self._cond:
			self._writer_active = False
			self._cond.notify_all()


class AlphaPoolCoordinatorActor(AlphaPoolActorBase):
	"""Authoritative AlphaPool coordinator.

	The coordinator directly owns the pool state and handles all
	mutations (``update_pool``, ``load_state``, …).

	Remote Ray workers (``AlphaPoolWorkerActor`` instances) are read-only
	evaluators.  Each worker keeps a lazily-synchronised snapshot of
	the pool; the snapshot is refreshed before every evaluation.
	Evaluation is dispatched to workers one factor at a time through
	a worker pool.
	"""

	def __init__(
		self,
		*,
		workers: Optional[List[Any]] = None,
		worker_count: Optional[int] = None,
		worker_num_cpus: int = 1,
		worker_memory: int | None = None,
		worker_resources: Optional[Dict[str, float]] = None,
		worker_label_selector: Optional[Dict[str, str]] = None,
		worker_init_kwargs: Optional[Dict[str, Any]] = None,
	):
		kw = dict(worker_init_kwargs or {})

		# ---- Head state ----
		self._returns = kw["returns"]
		self._data_idx = kw["data_idx"]
		self.pool = AlphaPoolState(capacity=int(kw["pool_capacity"]))

		# Feature filters
		self._feature_filter_params = dict(kw.get("feature_filter_params") or {})
		self._missing_threshold = float(self._feature_filter_params.get("missing_threshold", 0.3))
		self._low_std_threshold = float(self._feature_filter_params.get("low_std_threshold", 1e-4))

		# Quality gate
		self._quality_gate_mode = str(kw.get("quality_gate_mode", "smooth")).strip().lower()
		if self._quality_gate_mode not in ("smooth", "strict"):
			raise ValueError(
				f"quality_gate_mode must be 'smooth' or 'strict', got {self._quality_gate_mode!r}"
			)
		self._quality_gate_params = dict(kw.get("quality_gate_params") or {})

		# Trainer
		trainer_cfg = dict(kw.get("trainer_spec") or {})
		self._trainer_class_path = trainer_cfg["class_path"]
		self._trainer_params = dict(trainer_cfg.get("params") or {})
		trainer_cls = load_symbol(self._trainer_class_path)
		self.trainer = trainer_cls(**self._trainer_params)

		# Scorer
		scorer_cfg = dict(kw.get("scorer_spec") or {})
		self._scorer_class_path = scorer_cfg["class_path"]
		self._scorer_params = dict(scorer_cfg.get("params") or {})
		scorer_cls = load_symbol(self._scorer_class_path)
		self.scorer = scorer_cls(**self._scorer_params)

		# Persistence
		self._log_dir = kw.get("log_dir")
		self._auto_persist_state = bool(kw.get("auto_persist_state", True))

		# Pool tracking
		self._step_counter = 0
		self._best_pool_score: Optional[float] = None
		self._best_pool_step: Optional[int] = None
		self._feature_info_cache: Dict[str, Dict[str, Any]] = {}

		# Initial state
		initial_state_path = kw.get("initial_state_path")
		if initial_state_path:
			self._load_initial_state(initial_state_path)

		if self._log_dir:
			os.makedirs(self._log_dir, exist_ok=True)
			write_persist_manifest(self._log_dir, source="config")

		# ---- Workers ----
		if workers is None:
			if worker_count is None:
				raise ValueError("worker_count must be provided when workers is None")
			workers = self._spawn_workers(
				worker_count=int(worker_count),
				worker_num_cpus=int(worker_num_cpus),
				worker_memory=worker_memory,
				worker_resources=dict(worker_resources or {}),
				worker_label_selector=dict(worker_label_selector or {}),
				worker_init_kwargs=dict(kw),
			)
		if not workers:
			raise ValueError("workers must not be empty")

		self._workers = list(workers)
		self._lock = _AsyncWriterPrefRWLock()
		self._available_workers: asyncio.Queue[int] = asyncio.Queue()
		for idx in range(len(self._workers)):
			self._available_workers.put_nowait(idx)

		# ---- Snapshot management (incremental delta) ----
		self._snapshot_version = 0
		self._prev_entry_ids: List[str] = []
		self._delta_from_version: int = -1
		self._incremental_delta_ref: Any = None
		self._full_delta_ref: Any = None
		self._worker_versions = [-1 for _ in self._workers]
		self._publish_head_snapshot()

	# ------------------------------------------------------------------
	# Worker lifecycle
	# ------------------------------------------------------------------

	@staticmethod
	def _spawn_workers(
		*,
		worker_count: int,
		worker_num_cpus: int,
		worker_memory: int | None,
		worker_resources: Dict[str, float],
		worker_label_selector: Dict[str, str],
		worker_init_kwargs: Dict[str, Any],
	) -> List[Any]:
		if worker_count <= 0:
			raise ValueError("worker_count must be positive")

		worker_options: Dict[str, Any] = {"num_cpus": worker_num_cpus}
		if worker_memory is not None:
			worker_options["memory"] = int(worker_memory)
		if worker_resources:
			worker_options["resources"] = dict(worker_resources)
		if worker_label_selector:
			worker_options["label_selector"] = dict(worker_label_selector)

		evaluator_kwargs = dict(worker_init_kwargs or {})
		evaluator_kwargs["log_dir"] = None
		evaluator_kwargs["auto_persist_state"] = False
		evaluator_kwargs.pop("initial_state_path", None)

		RemoteWorker = ray.remote(**worker_options)(AlphaPoolWorkerActor)
		return [
			RemoteWorker.remote(**dict(evaluator_kwargs))
			for _ in range(worker_count)
		]

	async def shutdown(self) -> Dict[str, Any]:
		killed = 0
		for worker in self._workers:
			try:
				ray.kill(worker, no_restart=True)
				killed += 1
			except Exception:
				pass
		return {"ok": True, "workers": killed}

	# ------------------------------------------------------------------
	# Snapshot helpers
	# ------------------------------------------------------------------

	def _put_snapshot(self, payload: Any) -> Any:
		if not ray.is_initialized():
			return payload
		return ray.put(payload)

	def _publish_head_snapshot(self) -> None:
		self._snapshot_version += 1

		current_ids = self.pool.entry_factor_ids()
		prev_ids_set = set(self._prev_entry_ids)
		current_ids_set = set(current_ids)

		entry_manifest = [
			{"name": e.name, "factor_id": e.normalized().factor_id}
			for e in self.pool.entries
		]

		cache = self.pool.get_fit_cache()
		pool_metrics = copy_metrics(cache.get("metrics")) if cache else None
		importance = copy_importance(cache.get("importance")) if cache else None

		left_ids = [fid for fid in self._prev_entry_ids if fid not in current_ids_set]
		joined_ids = current_ids_set - prev_ids_set
		joined_entries = [
			e.to_snapshot_payload()
			for e in self.pool.entries
			if e.normalized().factor_id in joined_ids
		]

		self._incremental_delta_ref = self._put_snapshot({
			"entry_manifest": entry_manifest,
			"joined": joined_entries,
			"left_ids": left_ids,
			"pool_metrics": pool_metrics,
			"importance": importance,
		})
		self._delta_from_version = self._snapshot_version - 1

		self._full_delta_ref = self._put_snapshot({
			"entry_manifest": entry_manifest,
			"joined": self.pool.export_entries_snapshot(),
			"left_ids": [],
			"pool_metrics": pool_metrics,
			"importance": importance,
		})

		self._prev_entry_ids = list(current_ids)

	async def _ensure_worker_snapshot(self, idx: int, worker: Any) -> None:
		if self._worker_versions[idx] == self._snapshot_version:
			return
		if self._worker_versions[idx] == self._delta_from_version:
			ref = self._incremental_delta_ref
		else:
			ref = self._full_delta_ref
		await worker.apply_pool_delta.remote(self._snapshot_version, ref)
		self._worker_versions[idx] = self._snapshot_version

	# ------------------------------------------------------------------
	# Lock helpers
	# ------------------------------------------------------------------

	@asynccontextmanager
	async def _read_guard(self):
		await self._lock.acquire_read()
		try:
			yield
		finally:
			await self._lock.release_read()

	@asynccontextmanager
	async def _write_guard(self):
		await self._lock.acquire_write()
		try:
			yield
		finally:
			await self._lock.release_write()

	# ------------------------------------------------------------------
	# Evaluation dispatch
	# ------------------------------------------------------------------

	async def evaluate_batch(
		self,
		inputs: List[Any],
		*,
		names: Optional[List[str]] = None,
	) -> List[Dict[str, Any]]:
		"""Resolve inputs and dispatch individual factors to workers."""
		if not inputs:
			return []

		resolved, failures = self._resolve_inputs(inputs, names=names)

		async with self._read_guard():
			tasks = [
				self._evaluate_on_worker(name, series)
				for _, name, series in resolved
			]
			worker_results = await asyncio.gather(*tasks)

		results: List[Optional[Dict[str, Any]]] = [None] * len(inputs)
		for idx, failure_result in failures.items():
			results[idx] = failure_result
		for (idx, _, _), result in zip(resolved, worker_results):
			self._remember_feature_info(self._feature_info_from_payload(result))
			results[idx] = result

		return [item if item is not None else {} for item in results]

	async def _evaluate_on_worker(self, name: str, series: pd.Series) -> Dict[str, Any]:
		"""Lease one worker, evaluate one factor, return the worker."""
		idx = await self._available_workers.get()
		worker = self._workers[idx]
		try:
			await self._ensure_worker_snapshot(idx, worker)
			return await worker.evaluate_factor.remote(name, series)
		finally:
			self._available_workers.put_nowait(idx)

	# ------------------------------------------------------------------
	# Pool mutations
	# ------------------------------------------------------------------

	async def update_pool(
		self,
		inputs: List[Any],
		names: Optional[List[str]] = None,
	) -> Dict[str, Any]:
		async with self._write_guard():
			# 去重
			uniqued_inputs = []
			uniqued_names = []
			appeared_names = set()
			for idx, name in enumerate(names):
				if name is not None:
					appeared_names.add(name)
					uniqued_names.append(name)
					uniqued_inputs.append(inputs[idx])
			
			inputs = uniqued_inputs
			names = uniqued_names

			resolved, failures = self._resolve_inputs(inputs, names=names)
			factors = [(name, series) for _, name, series in resolved]
			

			filtered_factors, filtered_out_names = self._filter_factors_for_pool(factors)
			candidates = self._prepare_candidate_entries(filtered_factors)
			summary = self._apply_pool_update(
				candidates,
				update_kind="update_pool",
				candidate_names=[entry.name for entry in candidates],
				filtered_out_names=filtered_out_names,
			)

			if failures:
				summary["n_submitted"] = len(inputs)
				summary["n_resolved"] = len(resolved)
				summary["n_expr_failed"] = len(failures)
				summary["expr_failures"] = [failures[idx] for idx in sorted(failures)]

			self._publish_head_snapshot()
			return summary

	# ------------------------------------------------------------------
	# Pool queries
	# ------------------------------------------------------------------

	async def get_pool_status(self) -> Dict[str, Any]:
		async with self._read_guard():
			entries = list(self.pool.entries)

			# Retrieve importance from fit cache.
			importance_arr = None
			fit_cache = self.pool.get_fit_cache(entries)
			if fit_cache is not None:
				importance_arr = fit_cache.get("importance")

			entries_info = []
			for i, entry in enumerate(entries):
				normalized = entry.normalized()
				info: Dict[str, Any] = {
					"index": i,
					"name": normalized.name,
					"factor_id": normalized.factor_id,
					"feature_metrics": self._compute_factor_metrics(normalized.factor),
				}
				if importance_arr is not None and i < len(importance_arr):
					info["importance"] = float(importance_arr[i])
				fv = normalized.dense_values
				fv_finite = fv[np.isfinite(fv)]
				if len(fv_finite) >= 4:
					s = pd.Series(fv_finite)
					info["skew"] = float(s.skew())
					info["kurt"] = float(s.kurt())
				entries_info.append(info)

			pool_metrics = self._compute_pool_metrics()
			return {
				"size": len(entries),
				"capacity": self.pool.capacity,
				"step": self._step_counter,
				"best_pool_score": self._best_pool_score,
				"best_pool_step": self._best_pool_step,
				"pool_metrics": pool_metrics,
				"entries": entries_info,
			}

	async def get_pool_config(self) -> Dict[str, Any]:
		return {
			"pool_capacity": int(self.pool.capacity),
			"trainer": {
				"class_path": self._trainer_class_path,
				"params": dict(self._trainer_params),
			},
			"scorer": {
				"class_path": self._scorer_class_path,
				"params": dict(self._scorer_params),
			},
			"quality_gate": {
				"mode": self._quality_gate_mode,
				"params": dict(self._quality_gate_params),
			},
			"feature_filters": dict(self._feature_filter_params),
			"log_dir": self._log_dir,
			"auto_persist_state": self._auto_persist_state,
			"num_workers": len(self._workers),
		}

	async def get_data_range(self) -> Dict[str, str]:
		idx = self._data_idx
		return {
			"start": str(idx.min()),
			"end": str(idx.max()),
			"n_points": len(idx),
		}

	# ------------------------------------------------------------------
	# Persistence
	# ------------------------------------------------------------------

	async def save_state(self, directory: str) -> str:
		async with self._write_guard():
			os.makedirs(directory, exist_ok=True)
			self.pool.save(directory)
			write_actor_meta(
				os.path.join(directory, "actor_meta.json"),
				step=self._step_counter,
				best_pool_score=self._best_pool_score,
				best_pool_step=self._best_pool_step,
				extra={"pool_runtime": self.pool.export_fit_cache()},
			)
			return directory

	async def load_state(self, directory: str) -> None:
		async with self._write_guard():
			loaded = AlphaPoolState.load(directory, capacity=self.pool.capacity)
			self.pool.replace_entries(loaded.entries)
			meta = load_actor_meta(os.path.join(directory, "actor_meta.json"))
			if meta:
				self._step_counter = int(meta.get("step", 0))
				best_pool_score = meta.get("best_pool_score")
				self._best_pool_score = float(best_pool_score) if best_pool_score is not None else None
				best_pool_step = meta.get("best_pool_step")
				self._best_pool_step = int(best_pool_step) if best_pool_step is not None else None
				self.pool.restore_fit_cache(meta.get("pool_runtime"))
			self._publish_head_snapshot()

	# ------------------------------------------------------------------
	# Internal: Input resolution
	# ------------------------------------------------------------------

	def _resolve_inputs(
		self,
		inputs: List[Any],
		*,
		names: Optional[List[str]] = None,
	) -> Tuple[List[Tuple[int, str, pd.Series]], Dict[int, Dict[str, Any]]]:
		"""Normalize factor tuples or ExprEval payload refs into (idx, name, series)."""
		if not inputs:
			return [], {}

		if not self._inputs_are_payload_refs(inputs):
			resolved = [
				(idx, str(name), series)
				for idx, (name, series) in enumerate(inputs)
			]
			return resolved, {}

		resolved: List[Tuple[int, str, pd.Series]] = []
		failures: Dict[int, Dict[str, Any]] = {}
		fallback_names = list(names or [])
		payloads: Optional[List[Any]] = None

		try:
			payloads = ray.get(inputs) if inputs else []
		except Exception:
			payloads = None

		for idx, ref in enumerate(inputs):
			fallback_name = (
				str(fallback_names[idx])
				if idx < len(fallback_names) and fallback_names[idx] is not None
				else f"expr_{idx}"
			)
			try:
				payload = payloads[idx] if payloads is not None else ray.get(ref)
			except Exception as exc:
				failures[idx] = self._expr_eval_failure(
					fallback_name, f"expr_eval_error:{type(exc).__name__}",
				)
				continue
			try:
				name, series = self._factor_from_payload(payload, fallback_name)
			except Exception as exc:
				failures[idx] = self._expr_eval_failure(
					fallback_name, f"invalid_expr_payload:{type(exc).__name__}",
				)
				continue
			resolved.append((idx, name, series))

		return resolved, failures

	@staticmethod
	def _is_factor_input(item: Any) -> bool:
		return (
			isinstance(item, tuple)
			and len(item) == 2
			and isinstance(item[1], pd.Series)
		)

	@classmethod
	def _inputs_are_payload_refs(cls, inputs: Sequence[Any]) -> bool:
		if not inputs:
			return False
		factor_flags = [cls._is_factor_input(item) for item in inputs]
		if all(factor_flags):
			return False
		if any(factor_flags):
			raise TypeError("inputs must be either all factor pairs or all payload refs")
		return True

	@staticmethod
	def _series_from_payload(payload: Dict[str, Any]) -> pd.Series:
		raw = payload.get("series")
		if isinstance(raw, pd.Series):
			return raw.astype(float, copy=False)
		if not isinstance(raw, dict):
			raise ValueError("payload.series must be a pandas Series or JSON object")
		index_values = raw.get("index", [])
		try:
			index = pd.to_datetime(index_values)
		except Exception:
			index = pd.Index(index_values)
		values = raw.get("values", [])
		series = pd.Series(values, index=index, name=raw.get("name"))
		return series.astype(float, copy=False)

	@classmethod
	def _factor_from_payload(
		cls, payload: Dict[str, Any], fallback_name: str,
	) -> Tuple[str, pd.Series]:
		if not isinstance(payload, dict):
			raise ValueError("expression payload must be a JSON object")
		name = str(payload.get("expression") or fallback_name)
		return name, cls._series_from_payload(payload)

	# ------------------------------------------------------------------
	# Internal: Feature info cache
	# ------------------------------------------------------------------

	def _feature_info_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
		return {
			"name": str(payload["name"]),
			"valid": bool(payload["valid"]),
			"invalid_reason": payload["invalid_reason"],
			"feature_metrics": deepcopy(payload["feature_metrics"]),
		}

	def _remember_feature_info(self, feature_info: Dict[str, Any]) -> None:
		self._feature_info_cache[str(feature_info["name"])] = feature_info

	def review_feature_for_pool(self, feature_info: Optional[Dict[str, Any]], series: pd.Series) -> bool:
		if feature_info['invalid_reason'] is not None:
			return False
		
		if feature_info['feature_metrics']['compliance'] < 0.99:
			return False
		
		if series.quantile(0.99) <= series.quantile(0.01):
			return False
		
		# 将 feature 按照数值（min ~ max）分成 1000 桶，如果某个桶中的数的出现次数 > 10% 就 return False
		temp = series.dropna()
		buckets, _ = np.histogram(temp, bins=1000)
		for i in range(len(buckets)):
			if buckets[i] > 0.1 * len(temp):
				return False
		return True


	def _filter_factors_for_pool(
		self, factors: List[Tuple[str, pd.Series]],
	) -> Tuple[List[Tuple[str, pd.Series]], List[str]]:
		kept: List[Tuple[str, pd.Series]] = []
		filtered_out_names: List[str] = []
		for name, series in factors:
			feature_info = self._feature_info_cache.get(str(name))
			if feature_info is not None and not self.review_feature_for_pool(feature_info, series):
				filtered_out_names.append(str(name))
				continue
			kept.append((name, series))
		return kept, filtered_out_names

	# ------------------------------------------------------------------
	# Internal: Pool update engine
	# ------------------------------------------------------------------

	def _prepare_candidate_entries(
		self, factors: List[Tuple[str, pd.Series]],
	) -> List[PoolEntry]:
		entries: List[PoolEntry] = []
		for name, series in factors:
			aligned = self._validate_and_align_series(series, str(name))
			entries.append(PoolEntry(name=str(name), factor=aligned).normalized())
		return entries

	def _apply_pool_update(
		self,
		candidates: List[PoolEntry],
		*,
		update_kind: str,
		candidate_names: List[str],
		filtered_out_names: Optional[List[str]] = None,
	) -> Dict[str, Any]:
		pool_size_before = len(self.pool.entries)
		metrics_before = self._compute_pool_metrics()
		score_before = self._score_from_metrics(metrics_before)
		filtered_out_names = list(filtered_out_names or [])

		removed = self._joint_update(candidates)

		pool_size_after = len(self.pool.entries)
		if candidates:
			metrics_after, selected_importance = self._compute_metrics_for_entries(
				list(self.pool.entries),
			)
			self.pool.set_fit_cache(
				list(self.pool.entries),
				importance=selected_importance,
				metrics=metrics_after,
			)
		else:
			metrics_after = metrics_before
			selected_importance = None

		self._step_counter += 1
		if candidates:
			self._after_pool_update(
				removed, metrics_before, metrics_after,
				current_importance=selected_importance,
			)

		score_after = self._score_from_metrics(metrics_after)
		self._append_history_record(
			kind=update_kind,
			candidate_names=list(candidate_names),
			pool_updated=bool(candidates),
			pool_score_before=score_before,
			pool_score_after=score_after,
			n_candidates=len(candidates),
			n_removed=len(removed),
			n_filtered_out=len(filtered_out_names),
			filtered_out_names=filtered_out_names,
		)

		return {
			"pool_size_before": pool_size_before,
			"pool_size_after": pool_size_after,
			"pool_before": metrics_before,
			"pool_after": metrics_after,
			"n_candidates": len(candidates),
			"n_added": len(candidates),
			"n_removed": len(removed),
			"removed_names": [entry.name for entry in removed],
			"n_filtered_out": len(filtered_out_names),
			"filtered_out_names": filtered_out_names,
		}

	def _joint_update(
		self, candidates: List[PoolEntry],
	) -> List[PoolEntry]:
		if not candidates:
			return []

		current_entries = [entry.copy() for entry in self.pool.entries]
		new_candidates = [entry.normalized() for entry in candidates]
		all_entries = current_entries + new_candidates

		fit = self.trainer.fit_and_predict(
			all_entries, self._returns, self._data_idx, scorer=self.scorer,
		)
		importance = fit.importance
		if importance.size == 0:
			return []

		order = np.argsort(-np.abs(importance))[: self.pool.capacity]
		kept = set(int(i) for i in order.tolist())

		removed: List[PoolEntry] = []
		for idx, entry in enumerate(all_entries):
			if idx not in kept:
				removed.append(entry.copy())

		selected_entries = [all_entries[i].copy() for i in order]
		self.pool.replace_entries(selected_entries)
		return removed

	# ------------------------------------------------------------------
	# Internal: Metrics & scoring
	# ------------------------------------------------------------------

	def _compute_pool_metrics(self) -> Optional[Dict[str, Any]]:
		entries = list(self.pool.entries)
		cached = self.pool.get_fit_cache(entries)
		if cached is not None and (cached.get("metrics") is not None or not entries):
			return copy_metrics(cached.get("metrics"))
		metrics, importance = self._compute_metrics_for_entries(entries)
		self.pool.set_fit_cache(entries, importance=importance, metrics=metrics)
		return metrics

	# ------------------------------------------------------------------
	# Internal: Persistence helpers
	# ------------------------------------------------------------------

	def _load_initial_state(self, path: str) -> None:
		if os.path.isdir(path) and os.path.isfile(os.path.join(path, STATE_MANIFEST)):
			loaded = AlphaPoolState.load(path, capacity=self.pool.capacity)
			self.pool.replace_entries(loaded.entries)
			meta = load_actor_meta(os.path.join(path, "actor_meta.json"))
			if meta:
				self._step_counter = int(meta.get("step", 0))
				best_pool_score = meta.get("best_pool_score")
				self._best_pool_score = float(best_pool_score) if best_pool_score is not None else None
				best_pool_step = meta.get("best_pool_step")
				self._best_pool_step = int(best_pool_step) if best_pool_step is not None else None
				self.pool.restore_fit_cache(meta.get("pool_runtime"))
			return
		raise FileNotFoundError(
			f"initial_state_path must be a directory containing {STATE_MANIFEST}: {path}"
		)

	def _append_history_record(self, **fields: Any) -> None:
		if not self._log_dir:
			return
		score_before = fields.get("pool_score_before")
		score_after = fields.get("pool_score_after")
		delta = None
		if score_before is not None and score_after is not None:
			delta = float(score_after) - float(score_before)
		record = {
			"step": self._step_counter,
			"delta_score": delta,
			**fields,
		}
		append_pool_history_jsonl(self._log_dir, record)

	def _after_pool_update(
		self,
		dumped_entries: List[PoolEntry],
		metrics_before: Optional[Dict[str, Any]],
		metrics_after: Optional[Dict[str, Any]],
		*,
		current_importance: Optional[np.ndarray] = None,
	) -> None:
		if not self._log_dir:
			return

		step = self._step_counter
		entries = list(self.pool.entries)

		if self._auto_persist_state and entries:
			write_current_pool_snapshot(self._log_dir, self.pool, entries)

		score_after = self._score_from_metrics(metrics_after)
		if score_after is not None and entries:
			if self._best_pool_score is None or score_after > self._best_pool_score:
				self._best_pool_score = float(score_after)
				self._best_pool_step = int(step)
				write_best_pool_snapshot(
					self._log_dir, self.pool, entries,
					step=step, score=float(score_after),
				)

		if entries:
			feature_metrics_map = {
				name: info.get("feature_metrics")
				for name, info in self._feature_info_cache.items()
				if info.get("feature_metrics") is not None
			}
			write_dumped_features(
				self._log_dir,
				entries,
				current_importance,
				step=step,
				feature_metrics_map=feature_metrics_map,
			)

		write_actor_meta(
			os.path.join(self._log_dir, "actor_meta.json"),
			step=step,
			best_pool_score=self._best_pool_score,
			best_pool_step=self._best_pool_step,
			extra={"pool_runtime": self.pool.export_fit_cache()},
		)

	# ------------------------------------------------------------------
	# Static helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _score_from_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
		if metrics is None:
			return None
		score = metrics.get("score")
		return float(score) if score is not None else None
