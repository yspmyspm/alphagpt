import argparse
import os
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from config import ModelConfig, install_config, write_run_config_snapshot
from data_loader import AlphaDataLoader
from alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from vm import StackVM
from backtest import AlphaBacktest
from factors import FeatureEngineer
from helpers.eval_worker import (
	eval_single_formula,
	_init_worker,
	eval_formula_for_pool,
	_init_worker_pool,
)
from alphapool.pool_state import AlphaPoolState
from alphapool.batch_evaluator import AlphaPoolBatchEvaluator, BatchSummary, PoolMetrics
from alphapool.ensemble_trainer import LinearMeanStdEnsembleTrainer
from helpers.reward_metrics import compute_ic_metrics
from alphapool.pool_monitor import update_pool_monitoring
from utils.engine_base import RPNBasedAlphaEngine


class AlphaEngine(RPNBasedAlphaEngine):
	def __init__(self):
		"""
		Initialize AlphaGPT training engine.
		"""
		self.loader = AlphaDataLoader()
		self.loader.load_data()

		self.model = AlphaGPT().to(ModelConfig.DEVICE)

		# Standard optimizer
		self.opt = torch.optim.AdamW(self.model.parameters(), lr=ModelConfig.OPTIMIZER_LR)

		# Low-Rank Decay regularizer
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

		self.vm = StackVM()
		self.bt = AlphaBacktest(
			penalty=ModelConfig.BACKTEST_PENALTY,
			use_smooth_reward=ModelConfig.USE_SMOOTH_REWARD,
		)

		self.data_idx = self.loader.returns.index
		self.alpha_pool = AlphaPoolState(capacity=ModelConfig.ALPHA_POOL_SIZE)
		self.pool_batch_evaluator = AlphaPoolBatchEvaluator(
			returns=self.loader.returns,
			data_idx=self.data_idx,
			pool=self.alpha_pool,
			trainer=LinearMeanStdEnsembleTrainer(
				maxiter=ModelConfig.ENSEMBLE_MAXITER,
			),
			icir_missing_gamma=ModelConfig.ICIR_MISSING_GAMMA,
			icir_missing_eps=ModelConfig.ICIR_MISSING_EPS,
			missing_threshold=ModelConfig.ALPHA_MISSING_THRESHOLD,
			execute_fail_penalty=ModelConfig.EXECUTE_FAIL_PENALTY,
			missing_high_penalty=ModelConfig.MISSING_HIGH_PENALTY,
			low_std_penalty_base=ModelConfig.LOW_STD_PENALTY_BASE,
			compliance_fail_penalty=ModelConfig.COMPLIANCE_FAIL_PENALTY,
			max_corr_min_points=ModelConfig.ALPHA_MAX_CORR_MIN_POINTS,
		)
		self.pool_metrics_history: list = []

		self.best_score = -float('inf')
		self.best_formula = None
		self.best_pool_score = -float('inf')
		self.training_history = {
			'step': [],
			'avg_reward': [],
			'best_score': [],
			'stable_rank': []
		}
		self.op_kind_by_token_id = {
			self.model.op_start + i: spec[2]
			for i, spec in enumerate(self.model.op_specs)
		}
		self.run_dir = None
		self.start_step = 0

		resume_model_cfg = ModelConfig.RESUME_MODEL_CHECKPOINT
		resume_pool_cfg = ModelConfig.RESUME_ALPHA_POOL_PATH
		self.resume_model_checkpoint = self._normalize_optional_path(resume_model_cfg)
		self.resume_alpha_pool_path = self._normalize_optional_path(resume_pool_cfg)

		if self.resume_model_checkpoint:
			last_step = self.load_checkpoint(
				self.resume_model_checkpoint,
				load_alpha_pool=not bool(self.resume_alpha_pool_path),
			)
			self.start_step = max(0, int(last_step) + 1)

		if self.resume_alpha_pool_path:
			self._load_alpha_pool_from_path(self.resume_alpha_pool_path)

	@staticmethod
	def _normalize_optional_path(path: Optional[str]) -> Optional[str]:
		if path is None:
			return None
		s = str(path).strip()
		if not s:
			return None
		return os.path.abspath(os.path.expanduser(s))

	def _policy_gradient_step(self, log_probs, rewards: torch.Tensor):
		adv = (rewards - rewards.mean()) / (rewards.std() + ModelConfig.POLICY_ADVANTAGE_EPS)
		loss = 0
		for t in range(len(log_probs)):
			loss += -log_probs[t] * adv
		loss = loss.mean()

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

		if self.use_lord:
			self.lord_opt.step()

	def _log_step(self, step: int, avg_reward: float, pbar):
		postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}

		if self.use_lord and self.lord_rank_log_every > 0 and step % self.lord_rank_log_every == 0:
			stable_rank = self.rank_monitor.compute()
			postfix_dict['Rank'] = f"{stable_rank:.2f}"
			self.training_history['stable_rank'].append(stable_rank)

		self.training_history['step'].append(step)
		self.training_history['avg_reward'].append(avg_reward)
		self.training_history['best_score'].append(self.best_score)

		pbar.set_postfix(postfix_dict)

	def _eval_formula_batch(
		self,
		formulas,
		eval_indices,
		rewards: torch.Tensor,
		step: int,
		n_workers: int,
		*,
		pool_executor=None,
		factor_executor=None,
		precomputed_results=None,
	):
		"""Evaluate a finished-formula batch and write rewards back in place."""
		C_CYAN = "\033[1;36m"
		C_YELLOW = "\033[1;33m"
		C_DIM = "\033[2m"
		C_RESET = "\033[0m"

		if not formulas:
			return

		use_pool = ModelConfig.USE_ALPHA_POOL

		if use_pool:
			if precomputed_results is not None:
				pool_results = list(precomputed_results)
			elif pool_executor is not None:
				pool_results = list(pool_executor.map(eval_formula_for_pool, formulas))
			else:
				initargs = (
					self.loader.features,
					self.loader.returns,
					FeatureEngineer.INPUT_DIM,
					self.bt.use_smooth_reward,
					ModelConfig.ALPHA_MISSING_THRESHOLD,
					ModelConfig.LOW_STD_THRESHOLD,
					ModelConfig.LOW_STD_PENALTY_BASE,
				)
				with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker_pool, initargs=initargs) as ex:
					pool_results = list(ex.map(eval_formula_for_pool, formulas))

			summary: BatchSummary = self.pool_batch_evaluator.run_batch(
				pool_results,
				eval_indices,
				rewards,
				step=step,
				run_dir=self.run_dir,
				formula_to_str=self._formula_to_str,
			)

			if self.run_dir is not None:
				update_pool_monitoring(
					step=step,
					run_dir=self.run_dir,
					pool=self.alpha_pool,
					returns=self.loader.returns,
					data_idx=self.data_idx,
					trainer=self.pool_batch_evaluator.trainer,
					icir_missing_gamma=ModelConfig.ICIR_MISSING_GAMMA,
					icir_missing_eps=ModelConfig.ICIR_MISSING_EPS,
					bt=self.bt,
					history=self.pool_metrics_history,
					formula_to_str=self._formula_to_str,
				)

			self._print_pool_batch_summary(pool_results, summary, step)
			return

		if precomputed_results is not None:
			results_iter = list(precomputed_results)
		elif factor_executor is not None:
			results_iter = factor_executor.map(eval_single_formula, formulas)
		else:
			initargs = (
				self.loader.features,
				self.loader.returns,
				FeatureEngineer.INPUT_DIM,
				self.bt.use_smooth_reward,
				ModelConfig.EXECUTE_FAIL_PENALTY,
				ModelConfig.LOW_STD_PENALTY_BASE,
				ModelConfig.LOW_STD_THRESHOLD,
			)
			with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=initargs) as ex:
				results_iter = ex.map(eval_single_formula, formulas)
				results_iter = list(results_iter)

		batch_rows = []
		for idx, result in zip(eval_indices, results_iter):
			reward, score, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula = result
			rewards[idx] = reward
			batch_rows.append((score, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula))

		if not batch_rows:
			return

		round_best = max(batch_rows, key=lambda r: r[0])
		rs, d_icir, m_icir, o_ic, s_fin, s_di, s_hl, comp, f_best = round_best
		tqdm.write(
			f"{C_CYAN}[Top Factor] IC {o_ic:.4f} | dICIR {d_icir:.3f} | mICIR {m_icir:.3f} | "
			f"Compliance {comp:.3f} | {self._formula_to_str(f_best)}{C_RESET}"
		)
		if rs > self.best_score:
			self.best_score = rs
			self.best_formula = f_best
			tqdm.write(
				f"{C_YELLOW}[Best] IC {o_ic:.4f} | dICIR {d_icir:.3f} | mICIR {m_icir:.3f} | "
				f"{self._formula_to_str(f_best)}{C_RESET}"
			)
			p_archive = self.save_checkpoint(step=step, tag=f"best_step_{step + 1:06d}")
			p = self.save_checkpoint(step=step, tag="best")
			tqdm.write(f"  (saved checkpoint: {p}; archive: {p_archive})")

	def _print_pool_batch_summary(self, pool_results, summary: BatchSummary, step: int):
		C_CYAN = "\033[1;36m"
		C_YELLOW = "\033[1;33m"
		C_GREEN = "\033[1;32m"
		C_DIM = "\033[2m"
		C_RESET = "\033[0m"

		n_exec_fail = sum(1 for r in pool_results if not r.get("ok", False))
		n_missing = sum(1 for r in pool_results if r.get("missing_high"))
		n_low_std = sum(1 for r in pool_results if r.get("low_std"))
		n_valid = summary.n_candidates

		parts = [f"{len(pool_results)} EOS"]
		if n_valid: parts.append(f"{n_valid} valid")
		if n_exec_fail: parts.append(f"{n_exec_fail} exec_fail")
		if n_missing: parts.append(f"{n_missing} missing")
		if n_low_std: parts.append(f"{n_low_std} low_std")
		tqdm.write(f"{C_DIM}[Batch] {' | '.join(parts)}{C_RESET}")

		best_ic_info = None
		for r in pool_results:
			if not (r.get("ok") and "factor" in r and not r.get("missing_high") and not r.get("low_std")):
				continue
			ic = compute_ic_metrics(r["factor"], self.loader.returns)
			if ic is None:
				continue
			daily_ic, monthly_ic, overall_ic = ic
			d_std = daily_ic.std()
			m_std = monthly_ic.std()
			dicir = float(daily_ic.mean() / d_std) if d_std > 1e-8 else 0.0
			micir = float(monthly_ic.mean() / m_std) if m_std > 1e-8 else 0.0
			d_cov = float(np.isfinite(daily_ic.to_numpy(dtype=np.float64)).mean()) if len(daily_ic) > 0 else 0.0
			m_cov = float(np.isfinite(monthly_ic.to_numpy(dtype=np.float64)).mean()) if len(monthly_ic) > 0 else 0.0
			if best_ic_info is None or abs(overall_ic) > abs(best_ic_info[0]):
				best_ic_info = (overall_ic, dicir, micir, d_cov, m_cov, r["formula"])

		if best_ic_info:
			oic, dicir, micir, d_cov, m_cov, f = best_ic_info
			tqdm.write(
				f"{C_CYAN}[Top Factor] IC {oic:.4f} | dICIR {dicir:.3f} | mICIR {micir:.3f} | "
				f"cov {d_cov:.0%}/{m_cov:.0%} | {self._formula_to_str(f)}{C_RESET}"
			)

		mb: PoolMetrics | None = summary.pool_before
		ma: PoolMetrics | None = summary.pool_after
		sz_b = summary.pool_size_before
		sz_a = summary.pool_size_after

		if ma is not None and ma.score is not None:
			sa = ma.score
			pool_ic_str = (
				f"IC {ma.overall_ic:.4f} | dICIR {ma.daily_icir:.3f} | mICIR {ma.monthly_icir:.3f} | "
				f"cov {ma.daily_coverage:.0%}/{ma.monthly_coverage:.0%}"
			)
			if mb is not None and mb.score is not None and mb.score != 0:
				sb = mb.score
				delta = sa - sb
				pct = delta / abs(sb) * 100
				tqdm.write(
					f"{C_GREEN}[Pool] {sz_b}\u2192{sz_a} factors | "
					f"Score {sb:.4f}\u2192{sa:.4f} ({delta:+.4f}, {pct:+.1f}%) | {pool_ic_str}{C_RESET}"
				)
			else:
				tqdm.write(
					f"{C_GREEN}[Pool] {sz_a} factors | Score {sa:.4f} | {pool_ic_str}{C_RESET}"
				)

			if sa > self.best_pool_score:
				self.best_pool_score = sa
				if best_ic_info:
					self.best_formula = best_ic_info[5]
					self.best_score = sa
				tqdm.write(
					f"{C_YELLOW}[\u2605 Best Pool] Score {sa:.4f} | {sz_a} factors | {pool_ic_str}{C_RESET}"
				)
				p_archive = self.save_checkpoint(step=step, tag=f"best_step_{step + 1:06d}")
				p = self.save_checkpoint(step=step, tag="best")
				pool_path = self._save_best_pool_snapshot(step=step, score=sa)
				tqdm.write(f"  (saved checkpoint: {p}; archive: {p_archive}; best_pool: {pool_path})")
		elif sz_a > 0:
			tqdm.write(f"{C_GREEN}[Pool] {sz_a} factors (no score change){C_RESET}")

	def train(self):
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_dir = os.path.join(ModelConfig.LOGGING_DIR, ts)
		os.makedirs(self.run_dir, exist_ok=True)
		write_run_config_snapshot(self.run_dir)

		print("Starting AlphaGPT Training...")
		print(f"   Run directory: {self.run_dir}")
		if self.resume_model_checkpoint:
			print(f"   Resume checkpoint: {self.resume_model_checkpoint}")
		if self.resume_alpha_pool_path:
			print(f"   Resume alpha pool: {self.resume_alpha_pool_path}")
		print(f"   Start step: {self.start_step} / {ModelConfig.TRAIN_STEPS}")
		if self.use_lord:
			print("   LoRD Regularization enabled")
			print(f"   Decay keywords: {self.lord_decay_keywords}")
			print(f"   Rank monitor keywords: {self.lord_rank_monitor_keywords}")
			if self.lord_rank_log_every > 0:
				print(f"   Rank logging every {self.lord_rank_log_every} steps")
		if self.start_step > 0 and self.best_pool_score > -float('inf') and len(self.alpha_pool.entries) > 0:
			p = self._save_best_pool_snapshot(step=self.start_step - 1, score=self.best_pool_score)
			print(f"   Seed best pool snapshot: {p}")

		ckpt_every = int(ModelConfig.SAVE_CHECKPOINT_EVERY or 0)
		if ckpt_every > 0:
			print(f"   Checkpoints every {ckpt_every} steps -> {os.path.join(self.run_dir, 'checkpoints')}/")

		bs = ModelConfig.BATCH_SIZE
		no_eos_penalty = ModelConfig.NO_EOS_PENALTY
		n_workers = max(1, ModelConfig.EVAL_NUM_WORKERS)

		pbar = tqdm(range(self.start_step, ModelConfig.TRAIN_STEPS))
		use_pool = ModelConfig.USE_ALPHA_POOL

		##### start training
		for step in pbar:
			rewards = torch.full((bs,), no_eos_penalty, device=ModelConfig.DEVICE)
			if use_pool:
				initargs = (
					self.loader.features,
					self.loader.returns,
					FeatureEngineer.INPUT_DIM,
					self.bt.use_smooth_reward,
					ModelConfig.ALPHA_MISSING_THRESHOLD,
					ModelConfig.LOW_STD_THRESHOLD,
					ModelConfig.LOW_STD_PENALTY_BASE,
				)
				initializer = _init_worker_pool
				worker_fn = eval_formula_for_pool
			else:
				initargs = (
					self.loader.features,
					self.loader.returns,
					FeatureEngineer.INPUT_DIM,
					self.bt.use_smooth_reward,
					ModelConfig.EXECUTE_FAIL_PENALTY,
					ModelConfig.LOW_STD_PENALTY_BASE,
					ModelConfig.LOW_STD_THRESHOLD,
				)
				initializer = _init_worker
				worker_fn = eval_single_formula

			pending_indices = []
			pending_formulas = []
			pending_futures = []
			with ProcessPoolExecutor(max_workers=n_workers, initializer=initializer, initargs=initargs) as ex:
				inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
				log_probs = []
				alive_mask = torch.ones(bs, dtype=torch.bool, device=ModelConfig.DEVICE)
				temperature = max(ModelConfig.SAMPLING_TEMPERATURE_FLOOR, float(ModelConfig.GEN_TEMPERATURE))
				type_stacks = [[] for _ in range(bs)]
				formula_buffers = [[] for _ in range(bs)]

				for t in range(ModelConfig.MAX_FORMULA_LEN):
					logits, _, _ = self.model(inp)
					step_logits = logits / temperature
					remaining_steps = ModelConfig.MAX_FORMULA_LEN - (t + 1)

					for i in range(bs):
						if not alive_mask[i]:
							continue
						valid = torch.zeros(self.model.vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE)
						for token_id in range(self.model.vocab_size):
							if self._is_token_legal(token_id, type_stacks[i], remaining_steps=remaining_steps):
								valid[token_id] = True
						if not valid.any():
							valid[self.model.eos_token_id] = True
						step_logits[i] = step_logits[i].masked_fill(~valid, float('-inf'))

					dist = Categorical(logits=step_logits)
					action = dist.sample()
					step_log_prob = dist.log_prob(action) * alive_mask.float()
					log_probs.append(step_log_prob)

					for i in range(bs):
						if not alive_mask[i]:
							continue
						token_i = int(action[i].item())
						if token_i == self.model.eos_token_id:
							formula = list(formula_buffers[i])
							pending_indices.append(i)
							pending_formulas.append(formula)
							pending_futures.append(ex.submit(worker_fn, formula))
							continue
						formula_buffers[i].append(token_i)
						self._apply_token_to_stack(token_i, type_stacks[i])

					inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
					alive_mask = alive_mask & (action != self.model.eos_token_id)
					if not alive_mask.any():
						break

				if pending_futures:
					results = [f.result() for f in pending_futures]
					self._eval_formula_batch(
						pending_formulas,
						pending_indices,
						rewards,
						step,
						n_workers,
						precomputed_results=results,
					)
				else:
					C_DIM = "\033[2m"
					C_RESET = "\033[0m"
					tqdm.write(f"{C_DIM}[EOS] 0/{bs} — no complete expressions{C_RESET}")

			self._policy_gradient_step(log_probs, rewards)

			avg_reward = rewards.mean().item()
			self._log_step(step, avg_reward, pbar)

			if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
				p = self._save_step_checkpoint(step)
				tqdm.write(f"[checkpoint] step {step + 1} -> {p}")

		

		final_step = max(self.start_step, ModelConfig.TRAIN_STEPS) - 1
		final_path = self.save_checkpoint(step=final_step, tag="final")
		print(f"\nSaved final checkpoint: {final_path}")

		best_out = None
		if self.best_formula is not None:
			best_out = {
				"best_score": self.best_score,
				"best_pool_score": self.best_pool_score if self.best_pool_score > -float('inf') else None,
				"formula": self._formula_to_str(self.best_formula),
			}
		with open(os.path.join(self.run_dir, "best_meme_strategy.json"), "w", encoding="utf-8") as f:
			json.dump(best_out, f, ensure_ascii=False, indent=2)

		with open(os.path.join(self.run_dir, "training_history.json"), "w", encoding="utf-8") as f:
			json.dump(self.training_history, f, ensure_ascii=False, indent=2)

		vp = self._save_vocab_json()

		print("\nTraining completed!")
		print(f"  Artifacts: {self.run_dir}")
		print(f"  vocab.json: {vp}")
		if self.best_pool_score > -float('inf'):
			print(f"  Best pool score: {self.best_pool_score:.4f}")
		print(f"  Best score: {self.best_score:.4f}")
		if self.best_formula is not None:
			print(f"  Best formula: {self._formula_to_str(self.best_formula)}")
		else:
			print("  Best formula: (none)")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train AlphaGPT")
	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help="Path to config JSON file. Defaults to config.json in project root.",
	)
	args = parser.parse_args()
	install_config(args.config)

	import multiprocessing as mp
	mp.set_start_method(ModelConfig.MP_START_METHOD)
	eng = AlphaEngine()
	eng.train()
