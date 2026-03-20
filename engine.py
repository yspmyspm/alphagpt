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
		self.bt = AlphaBacktest(use_smooth_reward=getattr(ModelConfig, 'USE_SMOOTH_REWARD', True))
		
		self.best_score = -float('inf')
		self.best_formula = None
		self.training_history = {
			'step': [],
			'avg_reward': [],
			'best_score': [],
			'stable_rank': []
		}

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
			
			for _ in range(ModelConfig.MAX_FORMULA_LEN):
				logits, _, _ = self.model(inp)
				dist = Categorical(logits=logits)
				action = dist.sample()
				
				log_probs.append(dist.log_prob(action))
				tokens_list.append(action)
				inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
			
			seqs = torch.stack(tokens_list, dim=1)
			
			rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
			n_workers = max(1, getattr(ModelConfig, 'EVAL_NUM_WORKERS', 1))

			def _formula_to_str(formula):
				feat_offset = len(self.loader.feature_cols)
				return " ".join(
					self.loader.feature_cols[i] if i < feat_offset else self.model.vocab[i]
					for i in formula
				)

			formulas = [seqs[i].tolist() for i in range(bs)]
			initargs = (
				self.loader.features,
				self.loader.returns,
				FeatureEngineer.INPUT_DIM,
				self.bt.use_smooth_reward,
			)
			with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=initargs) as ex:
				for i, result in enumerate(ex.map(eval_single_formula, formulas)):
					reward, score, daily_icir, monthly_icir, overall_ic, s_finite, s_dist, s_halflife, compliance, formula = result
					rewards[i] = reward
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
