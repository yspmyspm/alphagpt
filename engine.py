import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor

from config import ModelConfig
from data_loader import AlphaDataLoader
from alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from vm import StackVM
from backtest import AlphaBacktest
from factors import FeatureEngineer
from helpers.eval_worker import eval_single_formula, _init_worker


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

	def train(self):
		print("Starting AlphaGPT Training...")
		if self.use_lord:
			print(f"   LoRD Regularization enabled")
			print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
		
		pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
		
		for step in pbar:
			bs = ModelConfig.BATCH_SIZE
			inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
			
			log_probs = []
			tokens_list = []
			alive_mask = torch.ones(bs, dtype=torch.bool, device=ModelConfig.DEVICE)
			temperature = max(1e-6, float(getattr(ModelConfig, 'GEN_TEMPERATURE', 1.0)))
			type_stacks = [[] for _ in range(bs)]
			
			for _ in range(ModelConfig.MAX_FORMULA_LEN):
				logits, _, _ = self.model(inp)
				step_logits = logits / temperature

				# 约束解码：对每个样本屏蔽非法 token
				for i in range(bs):
					if not alive_mask[i]:
						continue
					valid = torch.zeros(self.model.vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE)
					for token_id in range(self.model.vocab_size):
						if self._is_token_legal(token_id, type_stacks[i]):
							valid[token_id] = True
					# 极端情况下兜底，避免全 mask 导致 NaN
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
			
			unfinished_penalty = getattr(ModelConfig, 'UNFINISHED_PENALTY', -5.0)
			rewards = torch.full((bs,), unfinished_penalty, device=ModelConfig.DEVICE)
			n_workers = max(1, getattr(ModelConfig, 'EVAL_NUM_WORKERS', 1))

			def _formula_to_str(formula):
				feat_offset = len(self.loader.feature_cols)
				return " ".join(
					self.loader.feature_cols[i] if i < feat_offset else self.model.vocab[i]
					for i in formula
				)

			formulas = []
			eval_indices = []
			for i in range(bs):
				formula = seqs[i].tolist()
				if self.model.eos_token_id in formula:
					eos_pos = formula.index(self.model.eos_token_id)
					formula = formula[:eos_pos]
					formulas.append(formula)
					eval_indices.append(i)
			
			if formulas:
				initargs = (
					self.loader.features,
					self.loader.returns,
					FeatureEngineer.INPUT_DIM,
					self.bt.use_smooth_reward,
				)
				with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=initargs) as ex:
					for idx, result in zip(eval_indices, ex.map(eval_single_formula, formulas)):
						reward, score, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula = result
						rewards[idx] = reward
						if score > self.best_score:
							self.best_score = score
							self.best_formula = formula
							tqdm.write(f"[!] New King: Score {score:.3f} | Daily ICIR {daily_icir:.3f} | Monthly ICIR {monthly_icir:.3f} | Overall IC {overall_ic:.3f} | S_Finite {s_finite:.3f} | S_Dist {s_dist:.3f} | S_Halflife {s_halflife:.3f} | Compliance {compliance:.3f} | Formula {_formula_to_str(formula)}")
			
			# Normalize rewards
			adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
			
			loss = 0
			for t in range(len(log_probs)):
				loss += -log_probs[t] * adv
			
			loss = loss.mean()
			
			# Gradient step
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			
			# Apply Low-Rank Decay regularization
			if self.use_lord:
				self.lord_opt.step()
			
			# Logging
			avg_reward = rewards.mean().item()
			postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
			
			if self.use_lord and step % 100 == 0:
				stable_rank = self.rank_monitor.compute()
				postfix_dict['Rank'] = f"{stable_rank:.2f}"
				self.training_history['stable_rank'].append(stable_rank)
			
			self.training_history['step'].append(step)
			self.training_history['avg_reward'].append(avg_reward)
			self.training_history['best_score'].append(self.best_score)
			
			pbar.set_postfix(postfix_dict)

		# Save best formula
		with open("best_meme_strategy.json", "w") as f:
			json.dump(self.best_formula, f)
		
		# Save training history
		with open("training_history.json", "w") as f:
			json.dump(self.training_history, f)
		
		print("\nTraining completed!")
		print(f"  Best score: {self.best_score:.4f}")
		print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
	eng = AlphaEngine(use_lord_regularization=False)
	eng.train()
