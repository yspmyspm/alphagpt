import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from typing import Optional

import ray
import torch
from torch.distributions import Categorical
from tqdm import tqdm

_MAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MAIN_ROOT not in sys.path:
    sys.path.insert(0, _MAIN_ROOT)

from alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from configs.config import ModelConfig, install_config, write_run_config_snapshot

from utils.utils import *
from engine_base import RPNBasedAlphaEngine


class AlphaEngine(RPNBasedAlphaEngine):
    """Orchestrates the AlphaGPT training loop.

    Each training step: sample RPN formulas from the policy network -> dispatch
    async evaluation via ExprEval (feature computation) -> score formulas through
    AlphaPool (IC / ICIR metrics + diversity) -> compute shaped rewards ->
    update the policy with REINFORCE.
    """

    def __init__(self):
        self._connect_ray_services()

        token_catalog = dict(self.capabilities["token_catalog"])
        operator_specs = list(self.capabilities["operator_specs"])
        ts_parameters = list(self.capabilities["ts_parameters"])
        self._configure_rpn_schema(
            token_catalog=token_catalog,
            operator_specs=operator_specs,
            ts_parameters=ts_parameters,
        )
        ModelConfig.INPUT_DIM = len(self.feature_tokens)

        self.model = AlphaGPT(
            vocab_size=self.vocab_size,
        ).to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=ModelConfig.OPTIMIZER_LR)

        self.use_lord = bool(ModelConfig.USE_LORD_REGULARIZATION)
        self.lord_decay_keywords = list(ModelConfig.LORD_DECAY_KEYWORDS)
        self.lord_rank_monitor_keywords = list(ModelConfig.LORD_RANK_MONITOR_KEYWORDS)
        self.lord_rank_log_every = int(ModelConfig.LORD_RANK_LOG_EVERY)
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=float(ModelConfig.LORD_DECAY_RATE),
                num_iterations=int(ModelConfig.LORD_NUM_ITERATIONS),
                target_keywords=self.lord_decay_keywords,
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=self.lord_rank_monitor_keywords,
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self._refresh_pool_runtime_context()

        self.best_score = -float("inf")
        self.best_formula = None
        best_pool_score = self.pool_status.get("best_pool_score")
        self.best_pool_score = float(best_pool_score) if best_pool_score is not None else -float("inf")
        self.best_avg_reward = -float("inf")
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": [],
        }
        self.run_dir = None
        self.start_step = 0

        resume_model_cfg = ModelConfig.RESUME_MODEL_CHECKPOINT
        resume_pool_cfg = ModelConfig.RESUME_ALPHA_POOL_PATH
        self.resume_model_checkpoint = normalize_optional_path(resume_model_cfg)
        self.resume_alpha_pool_path = normalize_optional_path(resume_pool_cfg)

        if self.resume_model_checkpoint:
            ckpt_path = self._resolve_model_checkpoint_file(self.resume_model_checkpoint)
            last_step = self.load_checkpoint(ckpt_path)
            self.start_step = max(0, int(last_step) + 1)
            self.resume_model_checkpoint = ckpt_path

        if self.resume_alpha_pool_path:
            self.resume_alpha_pool_path = self._load_alpha_pool_from_path(self.resume_alpha_pool_path)

    def _policy_gradient_step(self, log_probs, rewards: torch.Tensor):
        """Run one policy-gradient update from the sampled rewards."""
        reward_std = rewards.std(unbiased=False)
        adv = (rewards - rewards.mean()) / (reward_std + ModelConfig.POLICY_ADVANTAGE_EPS)
        loss = 0
        for log_prob in log_probs:
            loss += -log_prob * adv
        loss = loss.mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.use_lord:
            self.lord_opt.step()

    def _log_step(self, step: int, avg_reward: float, pbar):
        postfix_dict = {"AvgRew": f"{avg_reward:.3f}", "BestScore": f"{self.best_score:.3f}"}

        if self.use_lord and self.lord_rank_log_every > 0 and step % self.lord_rank_log_every == 0:
            stable_rank = self.rank_monitor.compute()
            postfix_dict["Rank"] = f"{stable_rank:.2f}"
            self.training_history["stable_rank"].append(stable_rank)

        self.training_history["step"].append(step)
        self.training_history["avg_reward"].append(avg_reward)
        self.training_history["best_score"].append(self.best_score)

        pbar.set_postfix(postfix_dict)


    def _start_formula_eval_job(self, formulas, eval_indices):
        """Submit one finished formula batch to ExprEval."""
        tokenized_formulas = [self._formula_tokens(formula) for formula in formulas]
        formula_names = [self._formula_to_str(formula) for formula in formulas]

        # build expr_kwargs for ExprEval batch request
        expr_kwargs = {"expressions": tokenized_formulas}
        if getattr(self, "expr_data_path_overridden", False) and self.data_path:
            expr_kwargs["data_path"] = self.data_path

        try:
            expr_ref = self.expr_actor.evaluate_many_refs.remote(**expr_kwargs)
        except Exception as exc:
            return {
                "formulas": list(formulas),
                "eval_indices": list(eval_indices),
                "formula_names": formula_names,
                "expr_ref": None,
                "expr_error": exc,
            }
        return {
            "formulas": list(formulas),
            "eval_indices": list(eval_indices),
            "formula_names": formula_names,
            "expr_ref": expr_ref,
            "expr_error": None,
        }

    def _promote_expr_jobs(self, pending_expr_jobs, pending_pool_jobs, *, block: bool):
        """Move completed ExprEval jobs into the AlphaPool evaluate stage."""
        if not pending_expr_jobs:
            return
        while pending_expr_jobs:
            ready_jobs = []
            expr_refs = [job["expr_ref"] for job in pending_expr_jobs if job.get("expr_ref") is not None]
            if not expr_refs:
                ready_jobs = list(pending_expr_jobs)
                pending_expr_jobs.clear()
            else:
                ready_refs, _ = ray.wait(
                    expr_refs,
                    num_returns=1 if block else len(expr_refs),
                    timeout=None if block else 0,
                )
                if not ready_refs:
                    return
                ready_ref_set = set(ready_refs)
                remaining = []
                for job in pending_expr_jobs:
                    expr_ref = job.get("expr_ref")
                    if expr_ref is None or expr_ref in ready_ref_set:
                        ready_jobs.append(job)
                    else:
                        remaining.append(job)
                pending_expr_jobs[:] = remaining

            for job in ready_jobs:
                if job.get("expr_error") is not None:
                    pending_pool_jobs.append(
                        {
                            **job,
                            "payload_refs": None,
                            "pool_ref": None,
                            "pool_error": job["expr_error"],
                            "stage": "expr",
                        }
                    )
                    continue
                try:
                    payload_refs = ray.get(job["expr_ref"])
                except Exception as exc:
                    pending_pool_jobs.append(
                        {
                            **job,
                            "payload_refs": None,
                            "pool_ref": None,
                            "pool_error": exc,
                            "stage": "expr",
                        }
                    )
                    continue
                try:
                    pool_ref = self.pool_actor.evaluate_batch.remote(
                        inputs=payload_refs,
                        names=job["formula_names"],
                    )
                except Exception as exc:
                    pending_pool_jobs.append(
                        {
                            **job,
                            "payload_refs": payload_refs,
                            "pool_ref": None,
                            "pool_error": exc,
                            "stage": "pool_submit",
                        }
                    )
                    continue
                pending_pool_jobs.append(
                    {
                        **job,
                        "payload_refs": payload_refs,
                        "pool_ref": pool_ref,
                        "pool_error": None,
                        "stage": "pool_eval",
                    }
                )
            if not block:
                return

    def _reward_from_eval_result(self, result: dict) -> float | None:
        """Assemble the training reward from AlphaPool evaluation metrics."""
        if not result.get("valid"):
            return None
        feature_metrics = dict(result.get("feature_metrics") or {})
        diversity_metrics = dict(result.get("diversity_metrics") or {})
        pool_metrics_if_added = dict(result.get("pool_metrics_if_added") or {})
        combo_metrics = pool_metrics_if_added if pool_metrics_if_added else feature_metrics
        daily_icir = combo_metrics.get("daily_icir")
        monthly_icir = combo_metrics.get("monthly_icir")
        ic_score = combo_metrics.get("ic_score")
        if daily_icir is None or monthly_icir is None or ic_score is None:
            return None
        compliance = feature_metrics.get("compliance", 1.0)
        max_corr = diversity_metrics.get("max_corr", 0.0)
        try:
            compliance_val = float(compliance)
            daily_icir_val = float(daily_icir)
            monthly_icir_val = float(monthly_icir)
            ic_score_val = float(ic_score)
            max_corr_val = min(max(float(max_corr), 0.0), 1.0)
        except (TypeError, ValueError):
            return None
        reward_components_score = (
            abs(daily_icir_val) * float(ModelConfig.REWARD_W_DAILY)
            + abs(monthly_icir_val) * float(ModelConfig.REWARD_W_MONTHLY)
            + abs(ic_score_val) * float(ModelConfig.REWARD_W_IC)
        )
        return math.sqrt(1.0 - max_corr_val) * reward_components_score * compliance_val

    def _feature_perf_score_from_eval_result(self, result: dict) -> float | None:
        """Feature-only performance score (no diversity penalty), used for leaderboard ranking."""
        if not result.get("valid"):
            return None
        feature_metrics = dict(result.get("feature_metrics") or {})
        daily_icir = feature_metrics.get("daily_icir")
        monthly_icir = feature_metrics.get("monthly_icir")
        ic_score = feature_metrics.get("ic_score")
        if daily_icir is None or monthly_icir is None or ic_score is None:
            return None
        compliance = feature_metrics.get("compliance", 1.0)
        try:
            score = (
                abs(float(daily_icir)) * float(ModelConfig.REWARD_W_DAILY)
                + abs(float(monthly_icir)) * float(ModelConfig.REWARD_W_MONTHLY)
                + abs(float(ic_score)) * float(ModelConfig.REWARD_W_IC)
            ) * float(compliance)
        except (TypeError, ValueError):
            return None
        return score

    def _collect_pool_eval_jobs(
        self,
        pending_pool_jobs,
        rewards: torch.Tensor,
        step: int,
        *,
        update_pool: bool,
    ):
        """Collect AlphaPool evaluations, fill rewards, and optionally update the pool."""
        selected_payload_refs = []
        selected_names = []
        n_total = 0
        n_valid = 0
        all_results = []
        all_formulas = []
        for job in pending_pool_jobs:
            pool_error = job.get("pool_error")
            if pool_error is not None:
                for idx in job["eval_indices"]:
                    rewards[idx] = float(ModelConfig.NO_EOS_PENALTY)
                tqdm.write(f"[EvalBatchError] {type(pool_error).__name__}: {pool_error}")
                continue

            try:
                results = ray.get(job["pool_ref"])
            except Exception as exc:
                for idx in job["eval_indices"]:
                    rewards[idx] = float(ModelConfig.NO_EOS_PENALTY)
                tqdm.write(f"[PoolEvalError] {type(exc).__name__}: {exc}")
                continue

            n_total += len(results)
            n_valid += sum(1 for result in results if result.get("valid"))
            for idx, result in zip(job["eval_indices"], results):
                reward = self._reward_from_eval_result(result)
                rewards[idx] = (
                    float(reward) if reward is not None else float(ModelConfig.NO_EOS_PENALTY)
                )
            all_results.extend(results)
            all_formulas.extend(job["formulas"])

            if update_pool:
                for payload_ref, name, result in zip(
                    job["payload_refs"],
                    job["formula_names"],
                    results,
                ):
                    """Decide whether one evaluated factor should enter the step-end pool update."""
                    if result.get("valid"):
                        selected_payload_refs.append(payload_ref)
                        selected_names.append(name)


        self._print_factor_batch_summary(results, job["formulas"], step)

        if n_total > 0:
            tqdm.write(f"[Eval] {n_total} EOS | {n_valid} valid | {n_total - n_valid} invalid")

        if update_pool:
            if selected_payload_refs:
                summary = ray.get(
                    self.pool_actor.update_pool.remote(
                        inputs=selected_payload_refs,
                        names=selected_names,
                    )
                )
                self._print_pool_update_summary(summary, step)
            elif n_total > 0:
                tqdm.write("[Pool] 0 selected features for update")

    def _print_factor_batch_summary(self, results, formulas, step: int):
        accepted = [
            (result, formula)
            for result, formula in zip(results, formulas)
            if result.get("valid")
        ]
        if not accepted:
            tqdm.write("[Batch] 0 valid factors")
            return

        best_result, best_formula = max(
            accepted,
            key=lambda item: float(self._reward_from_eval_result(item[0]) or -float("inf")),
        )
        feature_metrics = dict(best_result.get("feature_metrics") or {})
        gain_metrics = dict(best_result.get("gain_metrics") or {})
        tqdm.write(
            "[Top Factor] "
            f"IC {feature_metrics.get('overall_ic', 0.0):.4f} | "
            f"dICIR {feature_metrics.get('daily_icir', 0.0):.3f} | "
            f"mICIR {feature_metrics.get('monthly_icir', 0.0):.3f} | "
            f"ic_score {feature_metrics.get('ic_score', 0.0):.3f} | "
            f"delta_ic {gain_metrics.get('overall_ic_delta', 0.0):+.4f} | "
            f"{self._formula_to_str(best_formula)}"
        )

        score = float(self._reward_from_eval_result(best_result) or -float("inf"))
        if score > self.best_score:
            self.best_score = score
            self.best_formula = best_formula
            tqdm.write(f"[Best] factor score {score:.4f} | {self._formula_to_str(best_formula)}")
        self._update_feature_leaderboard(results, step)

    def _print_pool_update_summary(self, summary: dict, step: int):
        before = summary.get("pool_before")
        after = summary.get("pool_after")
        size_before = summary.get("pool_size_before", 0)
        size_after = summary.get("pool_size_after", 0)
        n_candidates = summary.get("n_candidates", 0)
        n_removed = summary.get("n_removed", 0)

        tqdm.write(
            f"[Pool Update] candidates {n_candidates} | removed {n_removed} | size {size_before}->{size_after}"
        )

        if after and after.get("score") is not None:
            score_after = float(after["score"])
            pool_ic_str = (
                f"IC {after.get('overall_ic', 0.0):.4f} | "
                f"dICIR {after.get('daily_icir', 0.0):.3f} | "
                f"mICIR {after.get('monthly_icir', 0.0):.3f} | "
                f"cov {after.get('daily_coverage', 0.0):.0%}/{after.get('monthly_coverage', 0.0):.0%}"
            )
            if before and before.get("score") is not None and float(before["score"]) != 0.0:
                score_before = float(before["score"])
                delta = score_after - score_before
                pct = delta / abs(score_before) * 100.0
                tqdm.write(
                    f"[Pool] {size_before}->{size_after} factors | "
                    f"Score {score_before:.4f}->{score_after:.4f} ({delta:+.4f}, {pct:+.1f}%) | "
                    f"{pool_ic_str}"
                )
            else:
                tqdm.write(f"[Pool] {size_after} factors | Score {score_after:.4f} | {pool_ic_str}")

            if score_after > self.best_pool_score:
                self.best_pool_score = score_after
                self.best_score = max(self.best_score, score_after)
                manifest_path = self._save_pool_manifest(step=step, score=score_after)
                tqdm.write(f"[Best Pool] score {score_after:.4f} -> {manifest_path}")
        elif size_after > 0:
            tqdm.write(f"[Pool] {size_after} factors (no score available)")

    def train(self):
        """Run the full training loop, including sampling, eval, reward, and pool updates."""
        # ---- Stage 0: Setup – create run directory, snapshot configs, print banner ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(ModelConfig.LOGGING_DIR, ts)
        os.makedirs(self.run_dir, exist_ok=True)
        for sub in ("checkpoints", "best_alphapool"):
            os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)
        write_run_config_snapshot(self.run_dir)
        capabilities_path = self._write_capabilities_snapshot()
        token_catalog_path = self._save_token_catalog_json()
        ckpt_every = int(ModelConfig.SAVE_CHECKPOINT_EVERY or 0)

        self._print_training_start_banner(
            capabilities_path=capabilities_path,
            token_catalog_path=token_catalog_path,
            ckpt_every=ckpt_every,
        )

        batch_size = ModelConfig.BATCH_SIZE
        no_eos_penalty = float(ModelConfig.NO_EOS_PENALTY)
        pbar = tqdm(range(self.start_step, ModelConfig.TRAIN_STEPS))

        # ---- Main training loop ----
        for step in pbar:
            print("---------------- step ", step, " start ----------------")
            # -- Stage 1: Initialize per-step sampling state and async evaluation queues --
            rewards = torch.full((batch_size,), no_eos_penalty, device=ModelConfig.DEVICE)
            inp = torch.zeros((batch_size, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            log_probs = []
            alive_mask = torch.ones(batch_size, dtype=torch.bool, device=ModelConfig.DEVICE)
            temperature = max(ModelConfig.SAMPLING_TEMPERATURE_FLOOR, float(ModelConfig.GEN_TEMPERATURE))
            type_stacks = [[] for _ in range(batch_size)]
            formula_buffers = [[] for _ in range(batch_size)]
            pending_expr_jobs = []
            pending_pool_jobs = []

            # -- Stage 2: Autoregressive token sampling with on-the-fly async eval dispatch --
            for t in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                step_logits = logits / temperature
                remaining_steps = ModelConfig.MAX_FORMULA_LEN - (t + 1)

                forced_eos = [False] * batch_size
                for i in range(batch_size):
                    if not alive_mask[i]:
                        continue
                    valid, was_forced = self._build_valid_token_mask(
                        type_stacks[i], remaining_steps, t
                    )
                    if was_forced:
                        forced_eos[i] = True
                    step_logits[i] = step_logits[i].masked_fill(~valid, float("-inf"))

                dist = Categorical(logits=step_logits)
                action = dist.sample()
                step_log_prob = dist.log_prob(action) * alive_mask.float()
                log_probs.append(step_log_prob)

                finished_indices = []
                finished_formulas = []

                for i in range(batch_size):
                    # Finished formulas enter the async evaluation pipeline immediately.
                    if not alive_mask[i]:
                        continue
                    token_i = int(action[i].item())
                    if token_i == self.eos_token_id:
                        if forced_eos[i]:
                            rewards[i] = float(ModelConfig.NO_EOS_PENALTY)
                        else:
                            formula = list(formula_buffers[i])
                            finished_indices.append(i)
                            finished_formulas.append(formula)
                        continue
                    formula_buffers[i].append(token_i)
                    self._apply_token_to_stack(token_i, type_stacks[i])

                if finished_formulas:
                    pending_expr_jobs.append(
                        self._start_formula_eval_job(finished_formulas, finished_indices)
                    )

                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                alive_mask = alive_mask & (action != self.eos_token_id)
                self._promote_expr_jobs(pending_expr_jobs, pending_pool_jobs, block=False)
                if not alive_mask.any():
                    break

            # -- Stage 3: Drain remaining async ExprEval / AlphaPool jobs --
            self._promote_expr_jobs(pending_expr_jobs, pending_pool_jobs, block=True)
            if pending_pool_jobs:
                self._collect_pool_eval_jobs(
                    pending_pool_jobs,
                    rewards,
                    step,
                    update_pool=bool(ModelConfig.USE_ALPHA_POOL),
                )
            else:
                tqdm.write(f"[EOS] 0/{batch_size} no complete expressions")

            # -- Stage 4: Policy-gradient update (REINFORCE with baseline normalization) --
            self._policy_gradient_step(log_probs, rewards)

            # -- Stage 5: Logging, best-model tracking, periodic checkpointing --
            avg_reward = rewards.mean().item()
            self._log_step(step, avg_reward, pbar)

            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                p = self.save_checkpoint(step=step, tag="best", avg_reward=avg_reward)
                tqdm.write(f"[best_model] step {step + 1} avg_reward={avg_reward:.4f} -> {p}")

            if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
                p = self._save_step_checkpoint(step)
                tqdm.write(f"[checkpoint] step {step + 1} -> {p}")

        # ---- Post-loop: save final checkpoint, best strategy, and training artifacts ----
        final_step = max(self.start_step, ModelConfig.TRAIN_STEPS) - 1
        final_path = self.save_checkpoint(step=final_step, tag="final")

        best_out = None
        if self.best_formula is not None:
            best_out = {
                "best_score": self.best_score,
                "best_pool_score": self.best_pool_score if self.best_pool_score > -float("inf") else None,
                "formula": self._formula_to_str(self.best_formula),
            }
        with open(os.path.join(self.run_dir, "best_strategy.json"), "w", encoding="utf-8") as f:
            json.dump(best_out, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.run_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

        token_catalog_path = self._save_token_catalog_json()
        self._print_training_completion_summary(
            final_path=final_path,
            token_catalog_path=token_catalog_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlphaGPT")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON.",
    )
    args = parser.parse_args()
    install_config(args.config)
    engine = AlphaEngine()
    engine.train()

