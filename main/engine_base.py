"""Shared runtime, Ray service, and RPN grammar helpers for main engines."""

import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Optional

import ray
import torch

_MAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN_ROOT not in sys.path:
    sys.path.insert(0, _MAIN_ROOT)

from configs.config import ModelConfig
from remote_services import RayServiceBundle

from remote_services import (
    ALPHAPOOL_ACTOR_NAME,
    EXPR_EVAL_ACTOR_NAME,
    SERVICE_NAMESPACE,
)
from utils.utils import *

POOL_STATE_MANIFEST = "pool_state.json"


@dataclass(frozen=True)
class RPNTokenSchema:
    """Immutable vocabulary schema for RPN (Reverse Polish Notation) formula generation.

    Encodes the full token space – feature names, time-series parameters,
    operators, and the EOS sentinel – together with index ranges that let
    the engine classify any token_id by category in O(1).
    """

    feature_tokens: list[str]
    ts_param_tokens: list[str]
    operator_tokens: list[str]
    operator_specs: list[dict[str, Any]]
    ts_parameters: list[int]
    vocab: list[str]
    vocab_size: int
    eos_token: str
    eos_token_id: int
    ts_param_start: int
    ts_param_end: int
    op_start: int
    op_end: int
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def from_components(
        cls,
        *,
        token_catalog: dict[str, Any],
        operator_specs: list[dict[str, Any]],
        ts_parameters: list[int],
    ) -> "RPNTokenSchema":
        vocab = [str(token) for token in token_catalog["vocab"]]
        feature_tokens = [str(token) for token in token_catalog["feature_tokens"]]
        ts_param_tokens = [str(token) for token in token_catalog["ts_param_tokens"]]
        operator_tokens = [str(token) for token in token_catalog["operator_tokens"]]
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        return cls(
            feature_tokens=feature_tokens,
            ts_param_tokens=ts_param_tokens,
            operator_tokens=operator_tokens,
            operator_specs=[dict(spec) for spec in operator_specs],
            ts_parameters=[int(value) for value in ts_parameters],
            vocab=vocab,
            vocab_size=len(vocab),
            eos_token=str(token_catalog.get("eos_token", "<EOS>")),
            eos_token_id=int(token_catalog["eos_token_id"]),
            ts_param_start=int(token_catalog["ts_param_start"]),
            ts_param_end=int(token_catalog["ts_param_start"]) + len(ts_param_tokens),
            op_start=int(token_catalog["op_start"]),
            op_end=int(token_catalog["op_start"]) + len(operator_tokens),
            token_to_id=token_to_id,
            id_to_token={idx: token for token, idx in token_to_id.items()},
        )

    def to_token_catalog_payload(self) -> dict[str, Any]:
        return {
            "feature_tokens": list(self.feature_tokens),
            "ts_param_tokens": list(self.ts_param_tokens),
            "operator_tokens": list(self.operator_tokens),
            "vocab": list(self.vocab),
            "vocab_size": int(self.vocab_size),
            "eos_token": self.eos_token,
            "eos_token_id": int(self.eos_token_id),
            "ts_param_start": int(self.ts_param_start),
            "op_start": int(self.op_start),
            "token_to_id": dict(self.token_to_id),
            "id_to_token": {str(idx): token for idx, token in self.id_to_token.items()},
        }


class AlphaEngineBase:
    """Base runtime helpers for checkpoints, pool persistence, and run artifacts.

    Provides save/load for model checkpoints (with optional AlphaPool state)
    and utility methods for managing run directories and best-model snapshots.
    No dependency on Ray or the RPN grammar – those are added by subclasses.
    """

    def _write_capabilities_snapshot(self):
        if not self.run_dir:
            return None
        path = os.path.join(self.run_dir, "evaluator_capabilities.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.capabilities, f, ensure_ascii=False, indent=2)
        return path

    def _checkpoint_vocab(self) -> list[str]:
        return list(getattr(self, "vocab", []))

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

    def _get_pool_summary(self) -> Optional[dict]:
        """Fetch lightweight pool status from the AlphaPool actor (no disk I/O)."""
        pool_actor = getattr(self, "pool_actor", None)
        if pool_actor is None:
            return None
        try:
            return ray.get(pool_actor.get_pool_status.remote())
        except Exception:
            return None

    def _load_alpha_pool_from_path(self, path: str) -> str:
        resolved = self._resolve_alpha_pool_file(path)
        self._restore_alpha_pool_from_path(resolved)
        return resolved

    @staticmethod
    def _resolve_alpha_pool_file(path: str) -> str:
        if os.path.isdir(path):
            return path
        if os.path.isfile(path) and os.path.basename(path) == POOL_STATE_MANIFEST:
            return os.path.dirname(path)
        return os.path.abspath(os.path.expanduser(path))

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
            "model checkpoint not found under directory "
            "(expected checkpoints/latest/model.pt or step_*/model.pt): "
            f"{path}"
        )

    def _save_pool_manifest(self, step: int, score: float) -> str:
        """Write a lightweight pool manifest when best pool score improves.

        This does NOT save any model weights. AlphaPool manages its own
        persistence internally; main only records the pool composition here.
        """
        if not self.run_dir:
            raise RuntimeError("run_dir not set")
        d = os.path.join(self.run_dir, "best_alphapool")
        os.makedirs(d, exist_ok=True)
        meta = {
            "step": int(step),
            "score": float(score),
            "pool_status": self._get_pool_summary(),
        }
        path = os.path.join(d, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path

    def _save_step_checkpoint(self, step: int) -> str:
        tag = f"step_{step + 1:06d}"
        path = self.save_checkpoint(step=step, tag=tag)
        tag_dir = os.path.dirname(path)
        latest_dir = os.path.join(self._checkpoints_root(), "latest")
        if os.path.isdir(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(tag_dir, latest_dir)
        return path

    def _restore_alpha_pool_from_path(self, pool_path: str, *, checkpoint_path: str | None = None) -> bool:
        pool_actor = getattr(self, "pool_actor", None)
        if pool_actor is None:
            return False

        resolved = pool_path
        if checkpoint_path is not None and not os.path.isabs(resolved):
            resolved = os.path.normpath(os.path.join(os.path.dirname(checkpoint_path), resolved))
        if os.path.isfile(resolved) and os.path.basename(resolved) == POOL_STATE_MANIFEST:
            state_dir = os.path.dirname(resolved)
        else:
            state_dir = resolved
        ray.get(pool_actor.load_state.remote(directory=state_dir))
        return True

    def save_checkpoint(self, step: int, tag: str = "latest", **extra) -> str:
        """Save model weights, optimizer state, and training metadata.

        Any additional key-value pairs passed via **extra are merged into the
        checkpoint payload (e.g. avg_reward).
        """
        tag_dir = self._checkpoint_tag_dir(tag)
        path = os.path.join(tag_dir, "model.pt")
        payload = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score": self.best_score,
            "best_formula": self.best_formula,
            "best_pool_score": getattr(self, "best_pool_score", None),
            "best_avg_reward": getattr(self, "best_avg_reward", None),
            "training_history": getattr(self, "training_history", None),
            "vocab": self._checkpoint_vocab(),
            "capabilities": getattr(self, "capabilities", None),
            "pool_summary": self._get_pool_summary(),
        }
        payload.update(extra)
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        payload = torch.load(path, map_location=ModelConfig.DEVICE)
        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.opt.load_state_dict(payload["optimizer_state_dict"])
        self.best_score = payload.get("best_score", -float("inf"))
        self.best_formula = payload.get("best_formula")
        if hasattr(self, "best_pool_score"):
            setattr(self, "best_pool_score", payload.get("best_pool_score", getattr(self, "best_pool_score")))
        if hasattr(self, "best_avg_reward"):
            setattr(self, "best_avg_reward", float(payload.get("best_avg_reward", -float("inf"))))
        if hasattr(self, "training_history") and isinstance(payload.get("training_history"), dict):
            setattr(self, "training_history", payload["training_history"])
        return int(payload.get("step", 0))
    


    def _print_training_start_banner(
        self,
        *,
        capabilities_path: Optional[str],
        token_catalog_path: Optional[str],
        ckpt_every: int,
    ) -> None:
        print("Starting AlphaGPT Training...")
        print(f"   Run directory: {self.run_dir}")
        print("   Ray address: auto")
        print(f"   Service namespace: {SERVICE_NAMESPACE}")
        print(f"   Expr actor: {EXPR_EVAL_ACTOR_NAME}")
        print(f"   Pool actor: {ALPHAPOOL_ACTOR_NAME}")
        print(f"   Expr data path: {self.data_path}")
        if self.expr_data_path_overridden:
            print("   Expr data source: main override")
        else:
            print("   Expr data source: ExprEval default_data_path")
        print(
            "   Pool data range: "
            f"[{self.pool_data_range.get('start')}, {self.pool_data_range.get('end')}] "
            f"({self.pool_data_range.get('n_points')} points)"
        )
        if capabilities_path:
            print(f"   Evaluator capabilities: {capabilities_path}")
        print(f"   Token snapshot: {token_catalog_path}")
        print(
            f"   Feature tokens ({len(self.feature_tokens)}): "
            f"{self._format_token_line(self.feature_tokens)}"
        )
        print(
            f"   TS parameter tokens ({len(self.ts_param_tokens)}): "
            f"{self._format_token_line(self.ts_param_tokens)}"
        )
        print(
            f"   Operator tokens ({len(self.operator_tokens)}): "
            f"{self._format_token_line(self.operator_tokens)}"
        )
        print(f"   Vocab size: {self.vocab_size} | EOS token: {self.eos_token}")
        if self.resume_model_checkpoint:
            print(f"   Resume model: {self.resume_model_checkpoint}")
        if self.resume_alpha_pool_path:
            print(f"   Resume alpha pool: {self.resume_alpha_pool_path}")
        print(f"   Start step: {self.start_step} / {ModelConfig.TRAIN_STEPS}")
        if self.use_lord:
            print("   LoRD Regularization enabled")
            print(f"   Decay keywords: {self.lord_decay_keywords}")
            print(f"   Rank monitor keywords: {self.lord_rank_monitor_keywords}")
            if self.lord_rank_log_every > 0:
                print(f"   Rank logging every {self.lord_rank_log_every} steps")
        if ckpt_every > 0:
            print(
                f"   Checkpoints every {ckpt_every} steps -> "
                f"{os.path.join(self.run_dir, 'checkpoints')}/step_00000N/model.pt"
            )

    def _print_training_completion_summary(
        self,
        *,
        final_path: str,
        token_catalog_path: str,
    ) -> None:
        print(f"\nSaved final checkpoint: {final_path}")
        print("\nTraining completed!")
        print(f"  Artifacts: {self.run_dir}")
        print(f"  token_catalog.json: {token_catalog_path}")
        if self.best_pool_score > -float("inf"):
            print(f"  Best pool score: {self.best_pool_score:.4f}")
        print(f"  Best score: {self.best_score:.4f}")
        if self.best_formula is not None:
            print(f"  Best formula: {self._formula_to_str(self.best_formula)}")
        else:
            print("  Best formula: (none)")

    
    def _update_feature_leaderboard(self, results: list, step: int, top_n: int = 100):
        """Maintain a running top-N feature leaderboard (sorted by pure feature performance)."""
        if not self.run_dir:
            return
        path = os.path.join(self.run_dir, "feature_leaderboard.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                board = json.load(f)
        else:
            board = []

        for r in results:
            if not r.get("valid"):
                continue
            feature_score = self._feature_perf_score_from_eval_result(r)
            if feature_score is None or feature_score <= 0:
                continue
            feature_metrics = dict(r.get("feature_metrics") or {})
            quantiles = feature_metrics.pop("quantiles", None)
            board.append({
                "name": r.get("name", ""),
                "feature_score": float(feature_score),
                "importance": r.get("candidate_importance"),
                "step": step,
                "quantiles": quantiles,
                "feature_metrics": feature_metrics,
                "diversity_metrics": r.get("diversity_metrics", {}),
            })

        board.sort(key=lambda x: x["feature_score"], reverse=True)
        board = board[:top_n]
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_compact_number_lists(board))




class RayServiceAlphaEngineBase(AlphaEngineBase):
    """Adds Ray service connectivity on top of AlphaEngineBase.

    Connects to the ExprEval and AlphaPool Ray actors at startup and
    exposes convenience methods to refresh pool runtime context (config,
    status, data range).  Inherits checkpoint logic from AlphaEngineBase.
    """

    def _connect_ray_services(self) -> None:
        self.services = RayServiceBundle.connect()
        self.expr_actor = self.services.expr_actor
        self.pool_actor = self.services.pool_actor
        self.capabilities = self.services.capabilities
        self.data_path = self.services.data_path
        self.expr_data_path_overridden = self.services.expr_data_path_overridden

    def _refresh_pool_runtime_context(self) -> None:
        self.pool_config = ray.get(self.pool_actor.get_pool_config.remote())
        self.pool_status = ray.get(self.pool_actor.get_pool_status.remote())
        self.pool_data_range = ray.get(self.pool_actor.get_data_range.remote())


class RPNBasedAlphaEngine(RayServiceAlphaEngineBase):
    """Concrete engine layer that owns the RPN token schema and legality rules.

    Inherits Ray service access from RayServiceAlphaEngineBase and adds:
      - Token schema configuration (_configure_rpn_schema)
      - RPN stack-based legality checking (_is_token_legal, _build_valid_token_mask)
      - Formula serialization and vocab/token-catalog persistence

    Inheritance chain:
      AlphaEngineBase -> RayServiceAlphaEngineBase -> RPNBasedAlphaEngine
    """

    _MIN_WINDOW_BY_OP = {
        "ts_rank": 2,
        "ts_beta": 2,
        "ts_resid": 2,
        "ts_skew": 3,
        "ts_kurt": 4,
    }

    def _configure_rpn_schema(
        self,
        *,
        token_catalog: dict[str, Any],
        operator_specs: list[dict[str, Any]],
        ts_parameters: list[int],
    ) -> None:
        # Build the immutable schema from remote capability data
        schema = RPNTokenSchema.from_components(
            token_catalog=token_catalog,
            operator_specs=operator_specs,
            ts_parameters=ts_parameters,
        )
        # Unpack schema into instance attributes for fast access during decoding
        self.token_schema = schema
        self.feature_tokens = list(schema.feature_tokens)
        self.ts_param_tokens = list(schema.ts_param_tokens)
        self.operator_tokens = list(schema.operator_tokens)
        self.operator_specs = [dict(spec) for spec in schema.operator_specs]
        self.ts_parameters = list(schema.ts_parameters)
        self.vocab = list(schema.vocab)
        self.vocab_size = int(schema.vocab_size)
        self.token_to_id = dict(schema.token_to_id)
        self.id_to_token = dict(schema.id_to_token)
        self.eos_token = schema.eos_token
        self.eos_token_id = int(schema.eos_token_id)
        self.ts_param_start = int(schema.ts_param_start)
        self.ts_param_end = int(schema.ts_param_end)
        self.op_start = int(schema.op_start)
        self.op_end = int(schema.op_end)
        # Pre-build operator lookup maps (token_id -> kind / name) for legality checks
        self.op_kind_by_token_id = {
            self.op_start + i: str(spec["kind"])
            for i, spec in enumerate(self.operator_specs)
        }
        self.op_name_by_token_id = {
            self.op_start + i: str(spec["name"])
            for i, spec in enumerate(self.operator_specs)
        }

    def _require_token_schema(self) -> RPNTokenSchema:
        schema = getattr(self, "token_schema", None)
        if schema is None:
            raise RuntimeError("RPN token schema not configured")
        return schema

    def _formula_to_str(self, formula):
        return " ".join(self.vocab[i] for i in formula)

    def _formula_tokens(self, formula: list[int]) -> list[str]:
        return [self.vocab[token_id] for token_id in formula]

    def _save_token_catalog_json(self) -> str:
        schema = self._require_token_schema()
        path = os.path.join(self.run_dir, "token_catalog.json")
        payload = {
            "data_path": getattr(self, "data_path", None),
            "feature_tokens": list(self.feature_tokens),
            "ts_param_tokens": list(self.ts_param_tokens),
            "operator_tokens": list(self.operator_tokens),
            "vocab": list(self.vocab),
            "vocab_size": int(self.vocab_size),
            "eos_token": self.eos_token,
            "eos_token_id": int(self.eos_token_id),
            "token_catalog": schema.to_token_catalog_payload(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def _build_valid_token_mask(
        self, type_stack: list, remaining_steps: int, current_step: int
    ) -> tuple[torch.Tensor, bool]:
        device = ModelConfig.DEVICE
        valid = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        for token_id in range(self.vocab_size):
            if self._is_token_legal(token_id, type_stack, remaining_steps=remaining_steps, current_step=current_step):
                valid[token_id] = True
        if not valid.any():
            for rs in range(remaining_steps + 1, ModelConfig.MAX_FORMULA_LEN + 1):
                for token_id in range(self.vocab_size):
                    if self._is_token_legal(token_id, type_stack, remaining_steps=rs, current_step=current_step):
                        valid[token_id] = True
                if valid.any():
                    break
        if not valid.any():
            for token_id in range(self.vocab_size):
                if self._is_token_legal(token_id, type_stack, remaining_steps=ModelConfig.MAX_FORMULA_LEN, current_step=current_step):
                    valid[token_id] = True
                    break
        if not valid.any():
            forced = torch.zeros_like(valid)
            forced[self.eos_token_id] = True
            return forced, True
        return valid, False

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

    def _is_token_legal(self, token_id, type_stack, remaining_steps: int, current_step: int):
        max_stack_size = int(ModelConfig.MAX_DECODE_STACK_SIZE)

        if token_id == self.eos_token_id:
            if current_step < 3:
                return False
            return len(type_stack) == 1 and self._stack_entry_kind(type_stack[-1]) == "F"

        legal = False
        if token_id < self.ts_param_start:
            legal = True

        if self.ts_param_start <= token_id < self.ts_param_end:
            legal = len(type_stack) > 0 and self._stack_entry_kind(type_stack[-1]) == "F"

        if not legal and self.op_start <= token_id < self.op_end:
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
        if token_id < self.ts_param_start:
            type_stack.append("F")
            return
        if self.ts_param_start <= token_id < self.ts_param_end:
            param_value = None
            ts_params = getattr(self, "ts_parameters", None)
            if ts_params is not None:
                param_idx = int(token_id) - int(self.ts_param_start)
                if 0 <= param_idx < len(ts_params):
                    param_value = int(ts_params[param_idx])
            type_stack.append(("P", param_value) if param_value is not None else "P")
            return
        if self.op_start <= token_id < self.op_end:
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
