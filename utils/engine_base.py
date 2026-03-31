"""Shared training helpers for AlphaEngine."""
import glob
import json
import os
import shutil
from typing import Optional

import torch

from configs import ModelConfig


class AlphaEngineBase:
    """Base class for checkpointing, rollout, and training logs."""

    def _checkpoints_root(self) -> str:
        if not self.run_dir:
            raise RuntimeError("run_dir not set; train() must create the logging directory first.")
        d = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def _checkpoint_tag_dir(self, tag: str) -> str:
        d = os.path.join(self._checkpoints_root(), tag)
        os.makedirs(d, exist_ok=True)
        return d

    def _load_alpha_pool_from_path(self, path: str) -> str:
        resolved = self._resolve_alpha_pool_file(path)
        self._restore_alpha_pool_from_path(resolved)
        return resolved

    @staticmethod
    def _resolve_alpha_pool_file(path: str) -> str:
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for rel in (
                "alpha_pool.pkl",
                os.path.join("current_alphapool", "alpha_pool.pkl"),
                os.path.join("best_alphapool", "alpha_pool.pkl"),
            ):
                cand = os.path.join(path, rel)
                if os.path.isfile(cand):
                    return cand
        raise FileNotFoundError(
            f"alpha pool snapshot not found (expect a .pkl file or run dir with alpha_pool.pkl): {path}"
        )

    @staticmethod
    def _resolve_model_checkpoint_file(path: str) -> str:
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        for rel in (
            os.path.join("latest", "model.pt"),
            os.path.join("best_model", "model.pt"),
            os.path.join("final", "model.pt"),
        ):
            cand = os.path.join(path, rel)
            if os.path.isfile(cand):
                return cand
        step_pts = sorted(glob.glob(os.path.join(path, "checkpoints", "step_*", "model.pt")))
        if step_pts:
            return step_pts[-1]
        flat = os.path.join(path, "checkpoints", "latest.pt")
        if os.path.isfile(flat):
            return flat
        raise FileNotFoundError(
            f"model checkpoint not found under directory (expected checkpoints/latest/model.pt or step_*/model.pt): {path}"
        )

    def _save_best_pool_snapshot(self, step: int, score: float, *, checkpoint_model_path: str | None = None) -> str:
        if not self.run_dir:
            raise RuntimeError("run_dir not set")
        d = os.path.join(self.run_dir, "best_alphapool")
        os.makedirs(d, exist_ok=True)
        pool_path = os.path.join(d, "alpha_pool.pkl")
        meta_path = os.path.join(d, "meta.json")
        pool = getattr(self, "alpha_pool", None)
        if pool is None or not hasattr(pool, "save"):
            raise RuntimeError("alpha_pool is not initialized or not serializable")
        pool.save(pool_path)
        if checkpoint_model_path and os.path.isfile(checkpoint_model_path):
            shutil.copy2(checkpoint_model_path, os.path.join(d, "model.pt"))
        src_ckpt = None
        if checkpoint_model_path and self.run_dir:
            try:
                src_ckpt = os.path.relpath(checkpoint_model_path, start=self.run_dir)
            except ValueError:
                src_ckpt = checkpoint_model_path
        meta = {
            "step": int(step),
            "score": float(score),
            "pool_size": len(pool.entries),
            "source_checkpoint": src_ckpt,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._save_pool_feature_view_for_snapshot(step=step, subdir="best_alphapool")
        return pool_path

    def _save_pool_feature_view_for_snapshot(self, step: int, *, subdir: str) -> Optional[str]:
        """为 current/best alphapool 目录补充 manifest + 分布 + 因子值快照。"""
        if not self.run_dir:
            return None
        pool = getattr(self, "alpha_pool", None)
        if pool is None or not hasattr(pool, "entries") or len(pool.entries) == 0:
            return None
        formula_to_str = getattr(self, "_formula_to_str", None)
        if not callable(formula_to_str):
            return None
        loader = getattr(self, "loader", None)
        returns = getattr(loader, "returns", None)
        data_idx = getattr(self, "data_idx", None)
        evaluator = getattr(self, "pool_batch_evaluator", None)
        if returns is None or data_idx is None or evaluator is None:
            return None
        factors = pool.factor_series_list() if hasattr(pool, "factor_series_list") else [e.factor for e in pool.entries]
        if not factors:
            return None

        fit = evaluator.trainer.gfit(
            factors,
            returns,
            data_idx,
            icir_missing_gamma=evaluator.icir_missing_gamma,
            icir_missing_eps=evaluator.icir_missing_eps,
        )
        from alphapool.pool_feature_logging import save_pool_feature_snapshot

        return save_pool_feature_snapshot(
            self.run_dir,
            step,
            pool.entries,
            formula_to_str,
            returns,
            subdir=subdir,
            weights=fit.weights,
            use_step_subdir=False,
            include_feature_values=True,
        )

    def _save_step_checkpoint(self, step: int) -> str:
        tag = f"step_{step + 1:06d}"
        path = self.save_checkpoint(step=step, tag=tag)
        tag_dir = os.path.dirname(path)
        latest_dir = os.path.join(self._checkpoints_root(), "latest")
        if os.path.isdir(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(tag_dir, latest_dir)
        return path

    def _sync_current_alphapool(self, step: int) -> Optional[str]:
        pool = getattr(self, "alpha_pool", None)
        if pool is None or not hasattr(pool, "save") or not self.run_dir:
            return None
        d = os.path.join(self.run_dir, "current_alphapool")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "alpha_pool.pkl")
        pool.save(p)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"step": int(step), "pool_size": len(pool.entries)}, f, ensure_ascii=False, indent=2)
        return p

    def _save_best_model_by_avg_reward(self, step: int, avg_reward: float) -> str:
        if not self.run_dir:
            raise RuntimeError("run_dir not set")
        d = os.path.join(self.run_dir, "best_model")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "model.pt")
        pool_snapshot = None
        pool = getattr(self, "alpha_pool", None)
        if pool is not None and hasattr(pool, "save"):
            pool_snapshot = os.path.join(d, "alpha_pool.pkl")
            pool.save(pool_snapshot)
        payload = {
            "step": int(step),
            "avg_reward": float(avg_reward),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score": self.best_score,
            "best_formula": self.best_formula,
        }
        if hasattr(self, "best_pool_score"):
            payload["best_pool_score"] = getattr(self, "best_pool_score")
        if hasattr(self, "training_history"):
            payload["training_history"] = getattr(self, "training_history")
        if pool_snapshot:
            payload["alpha_pool_path"] = "alpha_pool.pkl"
        torch.save(payload, path)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"step": int(step), "avg_reward": float(avg_reward)}, f, ensure_ascii=False, indent=2)
        return path

    def _save_alpha_pool_snapshot_for_tag(self, tag: str) -> Optional[str]:
        pool = getattr(self, "alpha_pool", None)
        if pool is None or not hasattr(pool, "save"):
            return None
        pool_path = os.path.join(self._checkpoint_tag_dir(tag), "alpha_pool.pkl")
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
        tag_dir = self._checkpoint_tag_dir(tag)
        path = os.path.join(tag_dir, "model.pt")
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
        if hasattr(self, 'best_avg_reward'):
            payload['best_avg_reward'] = getattr(self, 'best_avg_reward')
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
        if hasattr(self, 'best_avg_reward'):
            setattr(self, 'best_avg_reward', float(payload.get('best_avg_reward', -float('inf'))))
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
	_MIN_WINDOW_BY_OP = {
		"ts_rank": 2,
		"ts_beta": 2,
		"ts_resid": 2,
		"ts_skew": 3,
		"ts_kurt": 4,
	}

	@staticmethod
	def _stack_entry_kind(entry):
		if isinstance(entry, tuple) and len(entry) >= 1:
			return entry[0]
		return entry

	@staticmethod
	def _stack_entry_param(entry):
		if isinstance(entry, tuple) and len(entry) >= 2 and entry[0] == "P":
			return int(entry[1])
		return None

	def _is_param_window_legal_for_op(self, token_id: int, param_value: int | None) -> bool:
		if param_value is None:
			return True
		op_name_by_token_id = getattr(self, "op_name_by_token_id", None)
		if not isinstance(op_name_by_token_id, dict):
			return True
		op_name = op_name_by_token_id.get(int(token_id))
		if op_name is None:
			return True
		min_window = self._MIN_WINDOW_BY_OP.get(op_name)
		if min_window is None:
			return True
		return int(param_value) >= int(min_window)

	def _minimum_tokens_to_finish(self, type_stack) -> int:
		f_count = sum(1 for t in type_stack if self._stack_entry_kind(t) == "F")
		p_count = sum(1 for t in type_stack if self._stack_entry_kind(t) == "P")
		if f_count <= 0:
			return 10**9
		return max(f_count - 1, p_count) + 1

	def _is_stack_finishable(self, type_stack, remaining_steps: int) -> bool:
		return self._minimum_tokens_to_finish(type_stack) <= max(0, int(remaining_steps))

	def _is_token_legal(self, token_id, type_stack, remaining_steps: int):
		max_stack_size = int(ModelConfig.MAX_DECODE_STACK_SIZE)

		if token_id == self.model.eos_token_id:
			return len(type_stack) == 1 and self._stack_entry_kind(type_stack[-1]) == "F"

		legal = False
		if token_id < self.model.ts_param_start:
			legal = True

		if self.model.ts_param_start <= token_id < self.model.ts_param_end:
			legal = len(type_stack) > 0 and self._stack_entry_kind(type_stack[-1]) == "F"

		if not legal and self.model.op_start <= token_id < self.model.op_end:
			kind = self.op_kind_by_token_id[token_id]
			if kind == "binary":
				legal = (
					len(type_stack) >= 2
					and self._stack_entry_kind(type_stack[-1]) == "F"
					and self._stack_entry_kind(type_stack[-2]) == "F"
				)
			if kind in ("unary_parameterized", "ts_unary"):
				legal = (
					len(type_stack) >= 2
					and self._stack_entry_kind(type_stack[-1]) == "P"
					and self._stack_entry_kind(type_stack[-2]) == "F"
				)
				if legal:
					param_value = self._stack_entry_param(type_stack[-1])
					legal = self._is_param_window_legal_for_op(token_id, param_value)
			if kind == "unary_parameterless":
				legal = len(type_stack) >= 1 and self._stack_entry_kind(type_stack[-1]) == "F"
			if kind == "ts_binary":
				legal = (
					len(type_stack) >= 3
					and self._stack_entry_kind(type_stack[-1]) == "P"
					and self._stack_entry_kind(type_stack[-2]) == "F"
					and self._stack_entry_kind(type_stack[-3]) == "F"
				)
				if legal:
					param_value = self._stack_entry_param(type_stack[-1])
					legal = self._is_param_window_legal_for_op(token_id, param_value)

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
			param_value = None
			ts_params = getattr(self.model, "ts_parameters", None)
			if ts_params is not None:
				param_idx = int(token_id) - int(self.model.ts_param_start)
				if 0 <= param_idx < len(ts_params):
					param_value = int(ts_params[param_idx])
			type_stack.append(("P", param_value) if param_value is not None else "P")
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
