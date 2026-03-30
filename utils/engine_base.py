"""AlphaEngine 的稳定子过程：检查点、词表导出、约束解码栈、rollout、策略梯度与日志。"""
import json
import os

import torch
from torch.distributions import Categorical
from tqdm import tqdm

from config import ModelConfig


class AlphaEngineBase:
    """由 AlphaEngine 继承；子类在 __init__ 中设置 self.model / self.loader 等属性。"""

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
