import os
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch
from tqdm import tqdm

from config import ModelConfig, write_run_config_snapshot
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
from alphapool.time_split import train_test_index_split
from alphapool.pool_state import AlphaPoolState
from alphapool.batch_evaluator import AlphaPoolBatchEvaluator
from alphapool.ensemble_trainer import LinearMeanStdEnsembleTrainer
from alphapool.pool_monitor import update_pool_monitoring
from utils.engine_base import AlphaEngineBase


class AlphaEngine(AlphaEngineBase):
	def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
		"""
		Initialize AlphaGPT training engine.

		Args:
			use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
			lord_decay_rate: Strength of LoRD regularization
			lord_num_iterations: Number of Newton-Schulz iterations per step
		"""
		self.loader = AlphaDataLoader()
		self.loader.load_data()

		self.model = AlphaGPT().to(ModelConfig.DEVICE)

		# Standard optimizer
		self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

		# Low-Rank Decay regularizer
		self.use_lord = use_lord_regularization
		if self.use_lord:
			self.lord_opt = NewtonSchulzLowRankDecay(
				self.model.named_parameters(),
				decay_rate=lord_decay_rate,
				num_iterations=lord_num_iterations,
				target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
			)
			self.rank_monitor = StableRankMonitor(
				self.model,
				target_keywords=["q_proj", "k_proj"]
			)
		else:
			self.lord_opt = None
			self.rank_monitor = None

		self.vm = StackVM()
		self.bt = AlphaBacktest(
			penalty=getattr(ModelConfig, 'BACKTEST_PENALTY', -5.0),
			use_smooth_reward=getattr(ModelConfig, 'USE_SMOOTH_REWARD', True),
		)

		self.train_idx, self.test_idx = train_test_index_split(
			self.loader.returns.index,
			train_years=getattr(ModelConfig, 'ALPHA_TRAIN_YEARS', 2.0),
		)
		self.alpha_pool = AlphaPoolState(capacity=getattr(ModelConfig, 'ALPHA_POOL_SIZE', 32))
		self.pool_batch_evaluator = AlphaPoolBatchEvaluator(
			returns=self.loader.returns,
			train_idx=self.train_idx,
			test_idx=self.test_idx,
			pool=self.alpha_pool,
			trainer=LinearMeanStdEnsembleTrainer(
				maxiter=int(getattr(ModelConfig, "ENSEMBLE_MAXITER", 400)),
			),
			icir_missing_gamma=ModelConfig.ICIR_MISSING_GAMMA,
			icir_missing_eps=ModelConfig.ICIR_MISSING_EPS,
			missing_threshold=getattr(ModelConfig, 'ALPHA_MISSING_THRESHOLD', 0.3),
			unfinished_penalty=getattr(ModelConfig, 'UNFINISHED_PENALTY', -5.0),
			fragment_eval=getattr(ModelConfig, 'ALPHA_POOL_FRAGMENT_EVAL', True),
		)
		self.pool_metrics_history: list = []

		self.best_score = -float('inf')
		self.best_formula = None
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

	def _eval_formulas_and_reward(self, seqs, rewards: torch.Tensor, bs: int, n_workers: int, step: int):
		"""对有 EOS 的样本做回测，写回 rewards，并打印 EOS / Round best / New King。"""
		formulas = []
		eval_indices = []
		for i in range(bs):
			formula = seqs[i].tolist()
			if self.model.eos_token_id in formula:
				eos_pos = formula.index(self.model.eos_token_id)
				formula = formula[:eos_pos]
				formulas.append(formula)
				eval_indices.append(i)

		n_eos = len(formulas)
		tqdm.write(f"[EOS] {n_eos}/{bs} expressions finished normally (EOS within max length)")

		c_new = "\033[1;33m"
		c_round = "\033[1;36m"
		c_reset = "\033[0m"

		if not formulas:
			return

		use_pool = getattr(ModelConfig, 'USE_ALPHA_POOL', True)

		if use_pool:
			initargs = (
				self.loader.features,
				self.loader.returns,
				FeatureEngineer.INPUT_DIM,
				self.bt.use_smooth_reward,
				getattr(ModelConfig, 'ALPHA_MISSING_THRESHOLD', 0.3),
			)
			
			with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker_pool, initargs=initargs) as ex:
				pool_results = list(ex.map(eval_formula_for_pool, formulas))
			
			self.pool_batch_evaluator.run_batch(pool_results, eval_indices, rewards)

			if self.run_dir is not None:
				update_pool_monitoring(
					step=step,
					run_dir=self.run_dir,
					pool=self.alpha_pool,
					returns=self.loader.returns,
					train_idx=self.train_idx,
					test_idx=self.test_idx,
					trainer=self.pool_batch_evaluator.trainer,
					icir_missing_gamma=ModelConfig.ICIR_MISSING_GAMMA,
					icir_missing_eps=ModelConfig.ICIR_MISSING_EPS,
					bt=self.bt,
					history=self.pool_metrics_history,
				)

			batch_rows = []
			for local_i, r in enumerate(pool_results):
				idx = eval_indices[local_i]
				sc = float(rewards[idx].item())
				ts = r.get('_test_score', 0.0) if isinstance(r, dict) else 0.0
				mc = r.get('_max_corr', 0.0) if isinstance(r, dict) else 0.0
				comp = r.get('compliance', 0.0) if isinstance(r, dict) else 0.0
				batch_rows.append((sc, ts, mc, comp, formulas[local_i]))

			if batch_rows:
				round_best = max(batch_rows, key=lambda x: x[0])
				rs, ts, mc, comp, f_best = round_best
				tqdm.write(
					f"{c_round}[Round best] Score {rs:.3f} | TestPerf {ts:.3f} | MaxCorr {mc:.3f} | "
					f"Compliance {comp:.3f} | Pool {len(self.alpha_pool.entries)} | "
					f"Formula {self._formula_to_str(f_best)}{c_reset}"
				)
				if rs > self.best_score:
					self.best_score = rs
					self.best_formula = f_best
					tqdm.write(
						f"{c_new}[!] New King: Score {rs:.3f} | TestPerf {ts:.3f} | MaxCorr {mc:.3f} | "
						f"Compliance {comp:.3f} | Formula {self._formula_to_str(f_best)}{c_reset}"
					)
					p = self.save_checkpoint(step=step, tag="best")
					tqdm.write(f"  (saved checkpoint: {p})")
		else:
			initargs = (
				self.loader.features,
				self.loader.returns,
				FeatureEngineer.INPUT_DIM,
				self.bt.use_smooth_reward,
			)
			with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=initargs) as ex:
				batch_rows = []
				for idx, result in zip(eval_indices, ex.map(eval_single_formula, formulas)):
					reward, score, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula = result
					rewards[idx] = reward
					batch_rows.append((score, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula))

				round_best = max(batch_rows, key=lambda r: r[0])
				rs, d_icir, m_icir, o_ic, s_fin, s_di, s_hl, comp, f_best = round_best
				tqdm.write(
					f"{c_round}[Round best] Score {rs:.3f} | Daily ICIR {d_icir:.3f} | Monthly ICIR {m_icir:.3f} | Overall IC {o_ic:.3f} | "
					f"S_Finite {s_fin:.3f} | S_Dist {s_di:.3f} | S_Halflife {s_hl:.3f} | Compliance {comp:.3f} | Formula {self._formula_to_str(f_best)}{c_reset}"
				)
				if rs > self.best_score:
					self.best_score = rs
					self.best_formula = f_best
					tqdm.write(
						f"{c_new}[!] New King: Score {rs:.3f} | Daily ICIR {d_icir:.3f} | Monthly ICIR {m_icir:.3f} | Overall IC {o_ic:.3f} | "
						f"S_Finite {s_fin:.3f} | S_Dist {s_di:.3f} | S_Halflife {s_hl:.3f} | Compliance {comp:.3f} | Formula {self._formula_to_str(f_best)}{c_reset}"
					)
					p = self.save_checkpoint(step=step, tag="best")
					tqdm.write(f"  (saved checkpoint: {p})")

	def train(self):
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_dir = os.path.join(ModelConfig.LOGGING_DIR, ts)
		os.makedirs(self.run_dir, exist_ok=True)
		write_run_config_snapshot(self.run_dir)

		print("Starting AlphaGPT Training...")
		print(f"   Run directory: {self.run_dir}")
		if self.use_lord:
			print("   LoRD Regularization enabled")
			print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")

		ckpt_every = int(getattr(ModelConfig, 'SAVE_CHECKPOINT_EVERY', 0) or 0)
		if ckpt_every > 0:
			print(f"   Checkpoints every {ckpt_every} steps -> {os.path.join(self.run_dir, 'checkpoints')}/")

		bs = ModelConfig.BATCH_SIZE
		unfinished_penalty = getattr(ModelConfig, 'UNFINISHED_PENALTY', -5.0)
		n_workers = max(1, getattr(ModelConfig, 'EVAL_NUM_WORKERS', 1))

		pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

		for step in pbar:
			seqs, log_probs = self._rollout_sequences(bs)
			rewards = torch.full((bs,), unfinished_penalty, device=ModelConfig.DEVICE)
			self._eval_formulas_and_reward(seqs, rewards, bs, n_workers, step)

			self._policy_gradient_step(log_probs, rewards)

			avg_reward = rewards.mean().item()
			self._log_step(step, avg_reward, pbar)

			if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
				p = self.save_checkpoint(step=step, tag="latest")
				tqdm.write(f"[checkpoint] step {step + 1} -> {p}")

		final_path = self.save_checkpoint(step=ModelConfig.TRAIN_STEPS - 1, tag="final")
		print(f"\nSaved final checkpoint: {final_path}")

		best_out = None
		if self.best_formula is not None:
			best_out = {
				"best_score": self.best_score,
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
		print(f"  Best score: {self.best_score:.4f}")
		if self.best_formula is not None:
			print(f"  Best formula: {self._formula_to_str(self.best_formula)}")
		else:
			print("  Best formula: (none)")


if __name__ == "__main__":
	import multiprocessing as mp
	mp.set_start_method("spawn")
	eng = AlphaEngine(use_lord_regularization=False)
	eng.train()
