"""
AlphaPool Ray coordinator deployment.

Usage:
  python serve.py --config config/config.json
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_ALPHAPOOL_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ALPHAPOOL_ROOT not in sys.path:
	sys.path.insert(0, _ALPHAPOOL_ROOT)

from configs.config import (
	ALPHAPOOL_ACTOR_NAME,
	ALPHAPOOL_NAMESPACE,
	load_alphapool_config_file,
	resolve_data_root,
	resolve_initial_pool_path,
	resolve_persist_dir,
)
from coordinator import AlphaPoolCoordinatorActor
from pool_state import STATE_MANIFEST


def _resolve_config_arg(config_path: Optional[str]) -> Optional[str]:
	if config_path:
		return os.path.abspath(os.path.expanduser(config_path))
	env = os.environ.get("ALPHAPOOL_CONFIG", "").strip()
	if env:
		return os.path.abspath(os.path.expanduser(env))
	return str(Path(__file__).resolve().parent / "config" / "config.json")


def _load_pool_config(config_path: Optional[str] = None) -> dict:
	path = _resolve_config_arg(config_path)
	if not os.path.isfile(path):
		raise FileNotFoundError(f"AlphaPool config not found: {path}")
	return load_alphapool_config_file(path)


def _build_actor_kwargs(cfg: dict, config_path_used: str) -> dict:
	"""Prepare constructor kwargs for AlphaPoolCoordinatorActor and AlphaPoolWorkerActor from config."""
	root = resolve_data_root(config_path_used, cfg)
	log_dir_resolved = resolve_persist_dir(config_path_used, cfg)

	# Load returns data
	rd = cfg["returns_data"]
	returns_dir = rd["returns_dir"]
	if not os.path.isabs(returns_dir):
		returns_dir = str(root / returns_dir)
	returns_path = os.path.join(returns_dir, rd["returns_filename"])
	ret_df = pd.read_feather(returns_path)
	if "_time" in ret_df.columns:
		ret_df.set_index("_time", inplace=True)
	ret_df.sort_index(inplace=True)
	returns = ret_df[rd["returns_column"]]

	return dict(
		returns=returns,
		data_idx=returns.index,
		pool_capacity=int(cfg["pool"]["alpha_pool_size"]),
		trainer_spec=cfg["trainer"],
		scorer_spec=cfg["scorer"],
		feature_filter_params=cfg["feature_filters"],
		quality_gate_mode=cfg["quality_gate"]["mode"],
		quality_gate_params=cfg["quality_gate"]["params"],
		log_dir=log_dir_resolved,
		auto_persist_state=bool(cfg["paths"]["auto_persist_state"]),
	)

def _get_named_actor(ray_module, actor_name: str, namespace: Optional[str]):
	try:
		if namespace:
			return ray_module.get_actor(actor_name, namespace=namespace)
		return ray_module.get_actor(actor_name)
	except ValueError:
		return None


def _wait_named_actor_gone(ray_module, actor_name: str, namespace: Optional[str], timeout_s: float = 30.0) -> None:
	deadline = time.time() + float(timeout_s)
	while time.time() < deadline:
		if _get_named_actor(ray_module, actor_name, namespace) is None:
			return
		time.sleep(0.5)
	raise TimeoutError(f"Timed out waiting for Ray actor {actor_name!r} to terminate")


def deploy_ray(
	config_path: Optional[str] = None,
	*,
	replace_existing: bool = True,
):
	"""Deploy AlphaPoolCoordinatorActor as a named Ray Actor on the cluster."""
	import ray

	resolved = _resolve_config_arg(config_path)
	cfg = _load_pool_config(resolved)
	deploy = cfg["deployment"]

	ray_address = deploy["ray_address"]
	actor_name = ALPHAPOOL_ACTOR_NAME
	namespace = ALPHAPOOL_NAMESPACE

	coord_cfg = deploy.get("coordinator") or {}
	worker_cfg = deploy.get("worker") or {}

	coord_num_cpus = int(coord_cfg.get("num_cpus", 0))
	coord_memory = coord_cfg.get("memory")
	coord_resources = coord_cfg.get("resources")
	coord_label_selector = coord_cfg.get("label_selector")

	num_workers = int(worker_cfg.get("count", 1))
	worker_num_cpus = int(worker_cfg.get("num_cpus", 1))
	worker_memory = worker_cfg.get("memory")
	worker_resources = worker_cfg.get("resources")
	worker_label_selector = worker_cfg.get("label_selector")

	if not ray.is_initialized():
		init_kwargs = {"address": ray_address}
		if namespace:
			init_kwargs["namespace"] = namespace
		ray.init(**init_kwargs)

	worker_kwargs = _build_actor_kwargs(cfg, config_path_used=resolved)

	# Resolve initial pool checkpoint path
	effective_pool_path = resolve_initial_pool_path(resolved, cfg)
	if effective_pool_path:
		if os.path.isdir(effective_pool_path) and os.path.isfile(os.path.join(effective_pool_path, STATE_MANIFEST)):
			worker_kwargs["initial_state_path"] = effective_pool_path
		elif os.path.isfile(effective_pool_path) and os.path.basename(effective_pool_path) == STATE_MANIFEST:
			worker_kwargs["initial_state_path"] = os.path.dirname(effective_pool_path)
		else:
			raise FileNotFoundError(
				f"Pool path must point to a directory containing {STATE_MANIFEST}: {effective_pool_path!r}"
			)

	# Handle existing actor
	existing = _get_named_actor(ray, actor_name, namespace)
	if existing is not None:
		if not replace_existing:
			print(f"[AlphaPool] Reusing existing Ray coordinator '{actor_name}'")
			return existing
		print(f"[AlphaPool] Replacing existing Ray coordinator '{actor_name}'")
		try:
			ray.get(existing.shutdown.remote())
		except Exception:
			pass
		try:
			ray.kill(existing, no_restart=True)
		except Exception:
			pass
		_wait_named_actor_gone(ray, actor_name, namespace)

	# Build coordinator Ray actor options
	coord_ray_options: Dict[str, Any] = {"num_cpus": coord_num_cpus}
	if coord_memory is not None and coord_memory != "":
		coord_ray_options["memory"] = int(coord_memory)
	if isinstance(coord_resources, dict) and coord_resources:
		coord_ray_options["resources"] = dict(coord_resources)
	if isinstance(coord_label_selector, dict) and coord_label_selector:
		coord_ray_options["label_selector"] = dict(coord_label_selector)

	RemoteActor = ray.remote(**coord_ray_options)(AlphaPoolCoordinatorActor)
	actor_options: Dict[str, Any] = {"name": actor_name, "lifetime": "detached"}
	if namespace:
		actor_options["namespace"] = namespace
	actor_handle = None
	try:
		actor_handle = RemoteActor.options(**actor_options).remote(
			worker_count=int(num_workers),
			worker_num_cpus=int(worker_num_cpus),
			worker_memory=None if worker_memory in (None, "") else int(worker_memory),
			worker_resources=(
				dict(worker_resources) if isinstance(worker_resources, dict) else None
			),
			worker_label_selector=(
				dict(worker_label_selector)
				if isinstance(worker_label_selector, dict) and worker_label_selector
				else None
			),
			worker_init_kwargs=dict(worker_kwargs),
		)
		status = ray.get(actor_handle.get_pool_status.remote())
	except Exception:
		if actor_handle is not None:
			try:
				ray.kill(actor_handle, no_restart=True)
			except Exception:
				pass
		raise

	if effective_pool_path:
		print(f"[AlphaPool] Loaded pool from {effective_pool_path}, size={status['size']}")

	idx = worker_kwargs["data_idx"]
	print(f"[AlphaPool] Ray coordinator '{actor_name}' deployed — "
		  f"capacity {worker_kwargs['pool_capacity']}, "
		  f"data range [{idx.min()}, {idx.max()}], "
		  f"coordinator num_cpus={coord_num_cpus}, "
		  f"workers={num_workers}, worker_num_cpus={worker_num_cpus}")
	print(f"[AlphaPool] Config: {resolved}")
	print(f"[AlphaPool] Namespace: {namespace}")
	print(f"[AlphaPool] Client usage: actor = ray.get_actor('{actor_name}', namespace='{namespace}')")
	return actor_handle


def main():
	parser = argparse.ArgumentParser(description="Deploy AlphaPool coordinator as a Ray Actor")
	parser.add_argument("--config", type=str, default=None, help="JSON config path")
	args = parser.parse_args()

	resolved = _resolve_config_arg(args.config)
	cfg = _load_pool_config(resolved)
	deploy_ray(config_path=resolved)
	print(f"[AlphaPool] Ray coordinator '{ALPHAPOOL_ACTOR_NAME}' deployed")
	print(f"[AlphaPool] Config: {resolved}")
	print(f"[AlphaPool] Namespace: {ALPHAPOOL_NAMESPACE}")
	print(f"[AlphaPool] Client usage: actor = ray.get_actor('{ALPHAPOOL_ACTOR_NAME}', namespace='{ALPHAPOOL_NAMESPACE}')")
	return 

if __name__ == "__main__":
	main()