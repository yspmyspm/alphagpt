import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_MAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MAIN_ROOT not in sys.path:
    sys.path.insert(0, _MAIN_ROOT)

from configs.config import ModelConfig


class NewtonSchulzLowRankDecay:
    """Low-Rank Decay (LoRD) regularizer using Newton-Schulz iteration.

    Approximates the polar factor (orthogonal component) of selected weight
    matrices via iterative Newton-Schulz steps and subtracts a fraction of it,
    pushing the weight spectrum toward lower effective rank without zeroing
    out singular values directly.  Applied after each optimizer step.
    """

    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
        self.params_to_decay = []

        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))

    @torch.no_grad()
    def step(self):
        for _, weight in self.params_to_decay:
            orig_dtype = weight.dtype
            x = weight.float()
            rows, cols = x.shape

            transposed = False
            if rows > cols:
                x = x.T
                transposed = True

            norm = x.norm() + 1e-8
            x = x / norm

            y = x
            ident = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
            for _ in range(self.num_iterations):
                gram = y.T @ y
                y = 0.5 * y @ (3.0 * ident - gram)

            if transposed:
                y = y.T

            weight.sub_(self.decay_rate * y.to(orig_dtype))


class StableRankMonitor:
    """Tracks the average stable rank (||W||_F^2 / sigma_max^2) of selected weight matrices.

    Stable rank is a continuous relaxation of matrix rank; monitoring it helps
    verify that the LoRD regularizer is effectively controlling capacity.
    """

    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        self.history = []

    @torch.no_grad()
    def compute(self):
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue

            weight = param.detach().float()
            singular_values = torch.linalg.svdvals(weight)
            stable_rank = (singular_values.norm() ** 2) / (singular_values[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())

        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))

    def forward(self, q, k):
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)

    def forward(self, x):
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.fc(x)


class MTPHead(nn.Module):
    """Multi-Task Prediction head with a learned soft router.

    Maintains `num_tasks` independent linear heads and a lightweight MLP router
    that produces per-token task weights.  The final logits are a weighted sum
    of all heads, allowing specialization across different formula structures.
    """

    def __init__(self, d_model, vocab_size, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(num_tasks)])
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks),
        )

    def forward(self, x):
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)
        task_outputs = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs, dim=1)
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)
        return weighted, task_probs


class LoopedTransformerLayer(nn.Module):
    """Single transformer layer whose weights are applied `num_loops` times (weight sharing).

    Instead of stacking distinct layers, the same attention + FFN block is
    iterated multiple times per forward pass.  This trades depth for parameter
    efficiency — the model gets deeper computation without extra parameters.
    """

    def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead
        self.qk_norm = QKNorm(d_model // nhead)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, is_causal=False):
        for _ in range(self.num_loops):
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=is_causal)
            x = x + self.dropout(attn_out)

            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)

        return x


class LoopedTransformer(nn.Module):
    """Weight-sharing transformer backbone composed of LoopedTransformerLayers.

    With `num_layers` layers each looped `num_loops` times, the effective depth
    is num_layers * num_loops while the parameter count scales only with
    num_layers.  This is the core sequence encoder used by AlphaGPT.
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    """Autoregressive policy network for generating RPN alpha formulas.

    Architecture: token embedding + learned positional embedding
    -> LoopedTransformer (weight-sharing causal transformer)
    -> RMSNorm -> MTPHead (multi-task soft-routed output)
    + a scalar critic head for value estimation.

    The model is trained with REINFORCE: it samples token sequences that form
    valid RPN expressions, and receives rewards from the AlphaPool evaluator.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
    ):
        super().__init__()
        self.d_model = int(ModelConfig.MODEL_D_MODEL)
        self.vocab_size = int(vocab_size)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, ModelConfig.MAX_FORMULA_LEN + 1, self.d_model))

        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=int(ModelConfig.MODEL_NHEAD),
            num_layers=int(ModelConfig.MODEL_NUM_LAYERS),
            dim_feedforward=int(ModelConfig.MODEL_DIM_FEEDFORWARD),
            num_loops=int(ModelConfig.MODEL_NUM_LOOPS),
            dropout=float(ModelConfig.MODEL_DROPOUT),
        )

        self.ln_f = RMSNorm(self.d_model)
        self.mtp_head = MTPHead(
            self.d_model,
            self.vocab_size,
            num_tasks=int(ModelConfig.MODEL_MTP_NUM_TASKS),
        )
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        _, seq_len = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)

        last_emb = x[:, -1, :]
        logits, task_probs = self.mtp_head(last_emb)
        value = self.head_critic(last_emb)
        return logits, value, task_probs
