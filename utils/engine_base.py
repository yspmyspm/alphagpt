"""Shared training helpers for AlphaEngine."""
import json
import os
import shutil
from typing import Optional

import torch

from config import ModelConfig


class AlphaEngineBase:
    """Base class for checkpointing, rollout, and training logs."""

    def _checkpoint_dir(self) -> str:
        if not self.run_dir:
            raise RuntimeError("run_dir not set; train() must create the logging directory first.")
        d = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self._checkpoint_dir(), name)

    def _checkpoint_pool_dir(self) -> str:
        d = os.path.join(self._checkpoint_dir(), "alpha_pool")
        os.makedirs(d, exist_ok=True)
        return d

    def _load_alpha_pool_from_path(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"alpha pool snapshot not found: {path}")
        self._restore_alpha_pool_from_path(path)

    def _save_best_pool_snapshot(self, step: int, score: float) -> str:
        if not self.run_dir:
            raise RuntimeError("run_dir not set")
        pool_path = os.path.join(self.run_dir, "best_alpha_pool.pkl")
        meta_path = os.path.join(self.run_dir, "best_alpha_pool_meta.json")
        pool = getattr(self, "alpha_pool", None)
        if pool is None or not hasattr(pool, "save"):
            raise RuntimeError("alpha_pool is not initialized or not serializable")
        pool.save(pool_path)
        meta = {
            "step": int(step),
            "score": float(score),
            "pool_size": len(pool.entries),
            "source_checkpoint": os.path.join("checkpoints", "best.pt"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return pool_path

    def _save_step_checkpoint(self, step: int) -> str:
        tag = f"step_{step + 1:06d}"
        path = self.save_checkpoint(step=step, tag=tag)
        latest_path = self._checkpoint_path("latest.pt")
        if os.path.abspath(path) != os.path.abspath(latest_path):
            shutil.copy2(path, latest_path)
        return path

    def _save_alpha_pool_snapshot_for_tag(self, tag: str) -> Optional[str]:
        pool = getattr(self, "alpha_pool", None)
        if pool is None or not hasattr(pool, "save"):
            return None
        pool_path = os.path.join(self._checkpoint_pool_dir(), f"{tag}.pkl")
        pool.save(pool_path)
        return pool_path

    def _restore_alpha_pool_from_path(self, pool_path: str, *, checkpoint_path: str | None = None) -> bool:
        pool = getattr(self, "alpha_pool", None)
        if pool is None:
            return False

        resolved = pool_path
        if checkpoint_path is not None and not os.path.isabs(resolved):
            resolved = os.path.normpath(os.path.join(os.path.dirname(checkpoint_path), resolved))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Alpha pool snapshot not found: {resolved}")

        pool_cls = pool.__class__
        if not hasattr(pool_cls, "load"):
            return False
        loaded = pool_cls.load(resolved, capacity=pool.capacity)
        if hasattr(pool, "replace_entries"):
            pool.replace_entries(loaded.entries)
        else:
            pool.entries = loaded.entries[: pool.capacity]

        evaluator = getattr(self, "pool_batch_evaluator", None)
        if evaluator is not None and hasattr(evaluator, "clear_cache"):
            evaluator.clear_cache()
        return True

    def save_checkpoint(self, step: int, tag: str = "latest") -> str:
        """Save model/optimizer state plus training progress and best metrics."""
        path = self._checkpoint_path(f"{tag}.pt")
        pool_snapshot = self._save_alpha_pool_snapshot_for_tag(tag)
        payload = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'best_score': self.best_score,
            'best_formula': self.best_formula,
        }
        if hasattr(self, 'best_pool_score'):
            payload['best_pool_score'] = getattr(self, 'best_pool_score')
        if hasattr(self, 'training_history'):
            payload['training_history'] = getattr(self, 'training_history')
        if pool_snapshot:
            payload['alpha_pool_path'] = os.path.relpath(pool_snapshot, start=os.path.dirname(path))
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str, *, load_alpha_pool: bool = True) -> int:
        """Load a checkpoint and return its saved training step."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        payload = torch.load(path, map_location=ModelConfig.DEVICE)
        self.model.load_state_dict(payload['model_state_dict'])
        if 'optimizer_state_dict' in payload:
            self.opt.load_state_dict(payload['optimizer_state_dict'])
        self.best_score = payload.get('best_score', -float('inf'))
        self.best_formula = payload.get('best_formula')
        if hasattr(self, 'best_pool_score'):
            setattr(self, 'best_pool_score', payload.get('best_pool_score', getattr(self, 'best_pool_score')))
        if hasattr(self, 'training_history') and isinstance(payload.get('training_history'), dict):
            setattr(self, 'training_history', payload['training_history'])
        pool_path = payload.get('alpha_pool_path')
        if load_alpha_pool and pool_path:
            self._restore_alpha_pool_from_path(pool_path, checkpoint_path=path)
        return int(payload.get('step', 0))

    def _formula_to_str(self, formula):
        feat_offset = len(self.loader.feature_cols)
        return " ".join(
            self.loader.feature_cols[i] if i < feat_offset else self.model.vocab[i]
            for i in formula
        )

    def _save_vocab_json(self) -> str:
        path = os.path.join(self.run_dir, "vocab.json")
        payload = {
            "tokens": list(self.model.vocab),
            "vocab_size": self.model.vocab_size,
            "eos_token": self.model.eos_token,
            "eos_token_id": self.model.eos_token_id,
            "feature_token_names": list(self.loader.feature_cols),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

class RPNBasedAlphaEngine(AlphaEngineBase):
	def _minimum_tokens_to_finish(self, type_stack) -> int:
		f_count = sum(1 for t in type_stack if t == "F")
		p_count = sum(1 for t in type_stack if t == "P")
		if f_count <= 0:
			return 10**9
		return max(f_count - 1, p_count) + 1

	def _is_stack_finishable(self, type_stack, remaining_steps: int) -> bool:
		return self._minimum_tokens_to_finish(type_stack) <= max(0, int(remaining_steps))

	def _is_token_legal(self, token_id, type_stack, remaining_steps: int):
		max_stack_size = int(ModelConfig.MAX_DECODE_STACK_SIZE)

		if token_id == self.model.eos_token_id:
			return len(type_stack) == 1 and type_stack[-1] == "F"

		legal = False
		if token_id < self.model.ts_param_start:
			legal = True

		if self.model.ts_param_start <= token_id < self.model.ts_param_end:
			legal = len(type_stack) > 0 and type_stack[-1] == "F"

		if not legal and self.model.op_start <= token_id < self.model.op_end:
			kind = self.op_kind_by_token_id[token_id]
			if kind == "binary":
				legal = len(type_stack) >= 2 and type_stack[-1] == "F" and type_stack[-2] == "F"
			if kind in ("unary_parameterized", "ts_unary"):
				legal = len(type_stack) >= 2 and type_stack[-1] == "P" and type_stack[-2] == "F"
			if kind == "unary_parameterless":
				legal = len(type_stack) >= 1 and type_stack[-1] == "F"
			if kind == "ts_binary":
				legal = (
					len(type_stack) >= 3
					and type_stack[-1] == "P"
					and type_stack[-2] == "F"
					and type_stack[-3] == "F"
				)

		if not legal:
			return False

		next_stack = list(type_stack)
		self._apply_token_to_stack(token_id, next_stack)
		if len(next_stack) > max_stack_size:
			return False
		return self._is_stack_finishable(next_stack, remaining_steps=remaining_steps)

	def _apply_token_to_stack(self, token_id, type_stack):
		if token_id < self.model.ts_param_start:
			type_stack.append("F")
			return
		if self.model.ts_param_start <= token_id < self.model.ts_param_end:
			type_stack.append("P")
			return
		if self.model.op_start <= token_id < self.model.op_end:
			kind = self.op_kind_by_token_id[token_id]
			if kind == "binary":
				type_stack.pop()
				type_stack.pop()
				type_stack.append("F")
			elif kind in ("unary_parameterized", "ts_unary"):
				type_stack.pop()
				type_stack.pop()
				type_stack.append("F")
			elif kind == "unary_parameterless":
				type_stack.pop()
				type_stack.append("F")
			elif kind == "ts_binary":
				type_stack.pop()
				type_stack.pop()
				type_stack.pop()
				type_stack.append("F")
