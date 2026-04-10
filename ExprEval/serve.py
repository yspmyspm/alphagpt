from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

_EXPREVAL_ROOT = Path(__file__).resolve().parent
if str(_EXPREVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPREVAL_ROOT))

from configs.config import load_deployment_config, resolve_config_path
from coordinator import ExpressionCoordinatorActor

import ray


def _get_named_actor(actor_name: str, namespace: Optional[str]):
    try:
        if namespace:
            return ray.get_actor(actor_name, namespace=namespace)
        return ray.get_actor(actor_name)
    except ValueError:
        return None


def _wait_named_actor_gone(actor_name: str, namespace: Optional[str], timeout_s: float = 30.0) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if _get_named_actor(actor_name, namespace) is None:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for Ray actor {actor_name!r} to terminate")

def deploy_ray(
    deploy: dict,
    *,
    config_path: Optional[str] = None,
    replace_existing: Optional[bool] = None,
):

    ray_address = deploy["ray_address"]
    actor_name = deploy["actor_name"]
    namespace = deploy["namespace"]
    worker_count = int(deploy["worker_count"])
    worker_options = {
        "num_cpus": int(deploy["worker_num_cpus"]),
        "memory": deploy.get("worker_memory"),
        "resources": deploy.get("worker_resources"),
        "label_selector": deploy.get("worker_label_selector"),
    }
    replace = deploy["replace_existing"] if replace_existing is None else bool(replace_existing)
    resolved = str(Path(config_path).resolve()) if config_path else None

    if not ray.is_initialized():
        init_kwargs = {"address": ray_address}
        if namespace:
            init_kwargs["namespace"] = namespace
        ray.init(**init_kwargs)

    existing = _get_named_actor(actor_name, namespace)
    if existing is not None:
        if not replace:
            print(f"[ExprEval] Reusing existing Ray actor '{actor_name}'")
            return existing
        print(f"[ExprEval] Replacing existing Ray actor '{actor_name}'")
        try:
            ray.kill(existing, no_restart=True)
        except Exception:
            pass
        _wait_named_actor_gone(actor_name, namespace)

    options = {"name": actor_name, "lifetime": "detached"}
    if namespace:
        options["namespace"] = namespace

    actor = ExpressionCoordinatorActor.options(**options).remote(
        worker_count=worker_count,
        worker_options=worker_options,
        config_path=resolved,
    )
    print(f"[ExprEval] Ray actor '{actor_name}' deployed")
    print(f"[ExprEval] Config: {resolved}")
    print(
        f"[ExprEval] Namespace: {namespace} | "
        f"workers={worker_count} | worker_num_cpus={worker_options['num_cpus']}"
    )
    if namespace:
        print(f"[ExprEval] Client usage: actor = ray.get_actor('{actor_name}', namespace='{namespace}')")
    else:
        print(f"[ExprEval] Client usage: actor = ray.get_actor('{actor_name}')")
    return actor


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy expression evaluator as a Ray actor")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to expression evaluator config JSON",
    )
    args = parser.parse_args()

    resolved = resolve_config_path(args.config)
    deploy = load_deployment_config(resolved)
    deploy_ray(deploy, config_path=resolved)

    if deploy["keep_alive"]:
        print("[ExprEval] Running (Ctrl+C to stop)...")

        def _sigint(sig, frame):
            print("\n[ExprEval] Shutting down...")
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint)
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
