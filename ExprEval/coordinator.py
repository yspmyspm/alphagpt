from __future__ import annotations

import asyncio
from typing import Any

import ray

from expression_evaluator import ExpressionEvaluator


class _WorkerLeasePool:
    """Lease evaluation workers so concurrent jobs stay bounded by worker count."""

    def __init__(self, workers: list[Any]):
        if not workers:
            raise ValueError("workers must not be empty")
        self._workers = list(workers)
        self._available_workers: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(self._workers)):
            self._available_workers.put_nowait(idx)

    async def acquire(self) -> tuple[int, Any]:
        idx = await self._available_workers.get()
        return idx, self._workers[idx]

    async def release(self, idx: int) -> dict[str, Any]:
        self._available_workers.put_nowait(int(idx))
        return {"ok": True, "worker_index": int(idx)}


WorkerLeasePoolActor = ray.remote(num_cpus=0)(_WorkerLeasePool)


@ray.remote(num_cpus=0)
def _run_expression_job(
    lease_pool: Any,
    expression,
    data_path: str | None = None,
):
    idx, worker = ray.get(lease_pool.acquire.remote())
    try:
        return ray.get(worker.evaluate_payload.remote(expression=expression, data_path=data_path))
    finally:
        ray.get(lease_pool.release.remote(idx))


class ExpressionCoordinator:
    """Dispatch expression evaluation to a bounded pool of Ray workers."""

    def __init__(
        self,
        *,
        workers: list[Any] | None = None,
        worker_count: int | None = None,
        worker_options: dict[str, Any] | None = None,
        config_path: str | None = None,
    ):
        if workers is None:
            if worker_count is None:
                raise ValueError("worker_count must be provided when workers is None")
            workers = self._spawn_workers(
                worker_count=int(worker_count),
                worker_options=dict(worker_options or {}),
                config_path=config_path,
            )
        if not workers:
            raise ValueError("workers must not be empty")
        self._workers = list(workers)
        self._lease_pool = WorkerLeasePoolActor.remote(self._workers)

    @staticmethod
    def _spawn_workers(
        *,
        worker_count: int,
        worker_options: dict[str, Any],
        config_path: str | None,
    ) -> list[Any]:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        options = {key: value for key, value in dict(worker_options).items() if value is not None}
        RemoteEvaluator = ray.remote(**options)(ExpressionEvaluator)
        return [
            RemoteEvaluator.remote(config_path=config_path)
            for _ in range(worker_count)
        ]

    def get_capabilities(self, data_path: str | None = None) -> dict[str, Any]:
        return ray.get(self._workers[0].get_capabilities.remote(data_path=data_path))

    def evaluate_ref(self, expression, data_path: str | None = None):
        return _run_expression_job.remote(
            self._lease_pool,
            expression,
            data_path,
        )

    def evaluate_many_refs(self, expressions, data_path: str | None = None) -> list[Any]:
        return [
            self.evaluate_ref(expression, data_path)
            for expression in expressions
        ]

    def shutdown(self) -> dict[str, Any]:
        killed = 0
        try:
            ray.kill(self._lease_pool, no_restart=True)
        except Exception:
            pass
        for worker in self._workers:
            try:
                ray.kill(worker, no_restart=True)
                killed += 1
            except Exception:
                pass
        return {"ok": True, "workers": killed}


ExpressionCoordinatorActor = ray.remote(num_cpus=0)(ExpressionCoordinator)
