import os
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch
from torch.distributions import Categorical
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


class AlphaEngine:
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
			trainer=LinearMeanStdEnsembleTrainer(),
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

	def _checkpoint_dir(self) -> str:
		if not self.run_dir:
			raise RuntimeError("run_dir not set; train() must create the logging directory first.")
		d = os.path.join(self.run_dir, "checkpoints")
		os.makedirs(d, exist_ok=True)
		return d

	def _checkpoint_path(self, name: str) -> str:
		return os.path.join(self._checkpoint_dir(), name)

	def save_checkpoint(self, step: int, tag: str = "latest") -> str:
		"""保存 model + optimizer + 训练进度与 best 指标。返回写入的文件路径。"""
		path = self._checkpoint_path(f"{tag}.pt")
		payload = {
			'step': step,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.opt.state_dict(),
			'best_score': self.best_score,
			'best_formula': self.best_formula,
		}
		torch.save(payload, path)
		return path

	def load_checkpoint(self, path: str) -> int:
		"""从检查点恢复，返回 checkpoint 中的 step。"""
		payload = torch.load(path, map_location=ModelConfig.DEVICE)
		self.model.load_state_dict(payload['model_state_dict'])
		self.opt.load_state_dict(payload['optimizer_state_dict'])
		self.best_score = payload.get('best_score', -float('inf'))
		self.best_formula = payload.get('best_formula')
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

	def _is_token_legal(self, token_id, type_stack):
		"""
		type_stack: list[str], 每项为 "F"(feature) 或 "P"(parameter)
		"""
		if token_id < self.model.ts_param_start:
			return True  # feature token

		if self.model.ts_param_start <= token_id < self.model.ts_param_end:
			# 参数 token：只允许接在 Feature 后，避免连续参数或起始参数
			return len(type_stack) > 0 and type_stack[-1] == "F"

		if self.model.op_start <= token_id < self.model.op_end:
			kind = self.op_kind_by_token_id[token_id]
			if kind == "binary":
				return len(type_stack) >= 2 and type_stack[-1] == "F" and type_stack[-2] == "F"
			if kind in ("unary_parameterized", "ts_unary"):
				return len(type_stack) >= 2 and type_stack[-1] == "P" and type_stack[-2] == "F"
			if kind == "unary_parameterless":
				return len(type_stack) >= 1 and type_stack[-1] == "F"
			if kind == "ts_binary":
				return (
					len(type_stack) >= 3
					and type_stack[-1] == "P"
					and type_stack[-2] == "F"
					and type_stack[-3] == "F"
				)
			return False

		# EOS：只允许完整表达式收束（栈中恰好 1 个 Feature）
		if token_id == self.model.eos_token_id:
			return len(type_stack) == 1 and type_stack[-1] == "F"

		return False

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

	def _rollout_sequences(self, bs: int):
		"""约束采样一整批公式，返回 token 序列与逐步 log_prob。"""
		inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
		log_probs = []
		tokens_list = []
		alive_mask = torch.ones(bs, dtype=torch.bool, device=ModelConfig.DEVICE)
		temperature = max(1e-6, float(getattr(ModelConfig, 'GEN_TEMPERATURE', 1.0)))
		type_stacks = [[] for _ in range(bs)]

		for _ in range(ModelConfig.MAX_FORMULA_LEN):
			logits, _, _ = self.model(inp)
			step_logits = logits / temperature

			for i in range(bs):
				if not alive_mask[i]:
					continue
				valid = torch.zeros(self.model.vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE)
				for token_id in range(self.model.vocab_size):
					if self._is_token_legal(token_id, type_stacks[i]):
						valid[token_id] = True
				if not valid.any():
					valid[self.model.eos_token_id] = True
				step_logits[i] = step_logits[i].masked_fill(~valid, float('-inf'))

			dist = Categorical(logits=step_logits)
			action = dist.sample()

			step_log_prob = dist.log_prob(action) * alive_mask.float()
			log_probs.append(step_log_prob)
			tokens_list.append(action)

			for i in range(bs):
				if not alive_mask[i]:
					continue
				token_i = int(action[i].item())
				if token_i == self.model.eos_token_id:
					continue
				self._apply_token_to_stack(token_i, type_stacks[i])

			inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
			alive_mask = alive_mask & (action != self.model.eos_token_id)

		seqs = torch.stack(tokens_list, dim=1)
		return seqs, log_probs

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
					fragment_eval=getattr(ModelConfig, 'ALPHA_POOL_FRAGMENT_EVAL', True),
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

	def _policy_gradient_step(self, log_probs, rewards: torch.Tensor):
		adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
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

		if self.use_lord and step % 100 == 0:
			stable_rank = self.rank_monitor.compute()
			postfix_dict['Rank'] = f"{stable_rank:.2f}"
			self.training_history['stable_rank'].append(stable_rank)

		self.training_history['step'].append(step)
		self.training_history['avg_reward'].append(avg_reward)
		self.training_history['best_score'].append(self.best_score)

		pbar.set_postfix(postfix_dict)

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
	eng = AlphaEngine(use_lord_regularization=False)
	eng.train()
