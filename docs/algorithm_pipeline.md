# AlphaGPT 算法与训练 Pipeline 详解

本文对应当前仓库实现（`engine.py` + `utils/engine_base.py` + `alphapool/*`），目标是把训练流程、奖励定义、约束解码、alpha pool 更新、checkpoint/resume 机制完整说明清楚。

## 1. 系统目标与核心思想

这套系统不是直接回归未来收益，而是做「**表达式生成**」：

1. 用自回归模型生成 RPN（逆波兰）表达式 token 序列。
2. 用 `StackVM` 执行表达式，得到因子序列。
3. 用 IC/ICIR + 合规性规则评估因子质量。
4. 用策略梯度更新生成模型，让模型更倾向于生成高质量表达式。
5. 可选地维护一个 alpha pool，把历史有效因子做线性组合并持续优化。

## 2. 数据层与输入约定

### 2.1 数据加载

由 `AlphaDataLoader` 完成：

- 特征文件：`ModelConfig.FEATHER_PATH`，通常是 `data/olhcv.feather`
- 收益文件：`ModelConfig.RETURNS_DIR/ModelConfig.RETURNS_FILENAME`，通常是 `returns/30min.feather`
- 若存在 `_time` 列会被设为索引；索引统一转 `datetime` 并排序。
- 不在加载阶段强制对齐特征和收益索引，评估时再取交集，避免 rolling 前段缺失被过早截断。

### 2.2 输入维度绑定

- `FeatureEngineer.INPUT_DIM` 与 `ModelConfig.INPUT_DIM` 在加载后被设为特征列数。
- 模型词表中的 `F0...F{N-1}` 由该维度决定。

## 3. 表达式语言（RPN）与 token 空间

表达式以 token id 序列表示，执行器为 `StackVM`。

### 3.1 token 分段

在 `AlphaGPT` 初始化时构造：

1. `Feature tokens`：`F0...F{INPUT_DIM-1}`
2. `TS parameter tokens`：`TS_1`, `TS_5`, ...（来自 `ModelConfig.TS_PARAMETERS`）
3. `Operator tokens`：来自 `helpers.myops.get_op_specs()`
4. `EOS token`：`<EOS>`

并记录边界：

- `ts_param_start`, `ts_param_end`
- `op_start`, `op_end`
- `eos_token_id`

### 3.2 操作符种类与栈签名

每个操作符具有 `(name, func, kind, arity)`，`kind` 决定参数栈签名：

- `binary`：`F F -> F`
- `unary_parameterless`：`F -> F`
- `unary_parameterized` / `ts_unary`：`F P -> F`
- `ts_binary`：`F F P -> F`

这里 `F` 表示序列因子，`P` 表示时间窗口参数。

## 4. 约束解码机制（重点）

训练时生成表达式不走“无约束采样”，而是按类型栈约束逐 token 过滤。

### 4.1 类型栈状态

`engine` 对 batch 内每个样本维护 `type_stack`：

- 采样特征 token：push `F`
- 采样参数 token：push `P`
- 采样 op token：按操作符类型弹栈/压栈，最终压回 `F`

### 4.2 token 合法性规则

`_is_token_legal(token_id, type_stack, remaining_steps)` 判断：

1. `EOS` 只在 `type_stack == [F]` 时合法（表达式已闭合）。
2. 参数 token 仅当栈顶是 `F` 时可采样。
3. 操作符需满足各自最小栈签名。
4. 应用该 token 后，栈长度不得超过 `MAX_DECODE_STACK_SIZE`。
5. 应用后状态必须在剩余步数内可收敛到可结束状态。

### 4.3 “可收敛”判定

`_minimum_tokens_to_finish(type_stack)` 估算“至少还需要多少 token 才能合法结束”：

- 记当前 `F` 个数为 `f_count`，`P` 个数为 `p_count`
- 若 `f_count <= 0`，不可结束（返回极大值）
- 否则最少需要：`max(f_count - 1, p_count) + 1`
  - `f_count - 1` 对应将多个 `F` 合并到 1 个 `F` 所需操作数
  - `p_count` 对应把参数消费掉至少需要的操作次数
  - `+1` 对应最终 `EOS`

只有当该值 `<= remaining_steps` 才允许该 token。

## 5. 模型结构

`AlphaGPT` 结构：

- token embedding + 可学习 positional embedding
- `LoopedTransformer`（2 层，每层 3 次循环）
  - RMSNorm
  - MultiheadAttention
  - SwiGLU FFN
- `MTPHead`：多头任务路由后输出 vocab logits
- `head_critic`：输出 value（当前训练 loop 未显式使用 value loss）

输出：`(logits, value, task_probs)`，训练 rollout 只用 `logits` 采样。

## 6. 单步训练流程（train step）

每个 `step` 的高层顺序：

1. 初始化 `rewards` 为 `NO_EOS_PENALTY`（默认所有样本先罚分）。
2. 在 `ProcessPoolExecutor` 中准备 worker（pool 模式与非 pool 模式初始化不同）。
3. 逐 token 解码：
   - 基于合法性掩码过滤 logits
   - 按温度采样 action
   - 累积每步 log_prob
   - 若某样本采到 `EOS`：
     - 立即把当前表达式提交到进程池评估（`ex.submit(...)`）
     - 该样本从 `alive_mask` 退出，不再继续解码
4. 全部解码结束后，收集 futures 结果并回写 batch reward。
5. 用策略梯度做一次更新。
6. 记录日志，按频率落盘 step checkpoint。

### 6.1 关键并行技巧：EOS 即时评测

当前实现不是“等 batch 等长结束后再统一评测”，而是：

- 某条序列一旦 `EOS`，马上提交评测任务到进程池；
- 主进程继续解码其他仍 alive 的样本；
- 最终统一 `future.result()` 回收结果。

这样短表达式不会因为长表达式未结束而阻塞评测队列。

## 7. 奖励与评估细节

### 7.1 执行层

`StackVM.execute(formula, features)`：

- 合法执行返回 `pd.Series`
- 任意栈错误/运算异常返回 `None`

### 7.2 非 pool 模式（直接因子奖励）

worker `eval_single_formula` 逻辑：

1. 执行失败：`EXECUTE_FAIL_PENALTY`
2. 标准差过低：按 `LOW_STD_PENALTY_BASE * (1 - std/1e-4)` 罚分
3. 正常：进入 `AlphaBacktest.evaluate`

### 7.3 IC 指标与加权得分

在对齐样本上计算：

- `overall_ic`
- `daily_ic` 序列 -> `daily_icir = mean/std`
- `monthly_ic` 序列 -> `monthly_icir = mean/std`

覆盖率惩罚：

- `daily_icir *= max(eps, coverage_daily)^gamma`
- `monthly_icir *= max(eps, coverage_monthly)^gamma`

加权分：

- `ic_score = overall_ic / (sqrt(daily_std * monthly_std + 1e-16) + 1e-8)`
- `score = 0.3*|daily_icir| + 0.3*|monthly_icir| + 0.4*|ic_score|`

### 7.4 合规分（compliance）

由三项平滑分相乘：

- `s_finite`：有限值比例分
- `s_dist`：分布形态分（偏度/峰度等）
- `s_halflife`：5 lag 自相关分

`compliance = s_finite * s_dist * s_halflife`

最终平滑奖励：

- `final_reward = penalty*(1-compliance) + good_reward*compliance`

## 8. Alpha Pool 机制

当 `USE_ALPHA_POOL=true` 时，单因子分不是直接 backtest 分，而是“候选对 pool 的边际贡献”。

### 8.1 pool 状态

`AlphaPoolState` 保存 `PoolEntry(formula, factor)`，容量 `ALPHA_POOL_SIZE`。
支持 pickle 落盘与加载。

### 8.2 候选 reward（每条 EOS）

`AlphaPoolBatchEvaluator._reward_single`：

1. 执行失败 / 缺失率过高 / 低方差，直接给惩罚。
2. 否则计算：
   - `max_c = max_abs_pearson(candidate, pool_factors)`
   - 对 `pool + candidate` 做 ensemble 拟合得到 `score`
   - `perf_reward = (1 - max_c) * score`
   - `reward = perf_reward * compliance`

这里同时鼓励：

- pool 总体表现提升（`score` 高）
- 与已有池内因子去冗余（`1-max_c`）
- 本身合规（`compliance`）

### 8.3 联合更新与裁剪

batch 内所有有效候选一起做一次 joint update：

1. 合并旧 pool + 新候选
2. 用 `LinearMeanStdEnsembleTrainer.gfit` 拟合权重
   - 参数化：`w = softmax(u)`
   - 优化器：`scipy.optimize.minimize(method='L-BFGS-B')`
   - 目标：最大化 `performance_score_on_subset(..., eval_mode='full_weighted')`
3. 按 `|w|` 排序，保留前 `capacity` 个条目
4. 被裁剪因子会写入 `pool_features_removed/...` 便于审计

### 8.4 pool 监控

每步调用 `update_pool_monitoring`：

- 用当前 pool 重新拟合组合权重
- 生成并保存组合预测序列
- 记录 `overall_ic / daily_icir / monthly_icir / weighted_score / final_reward`
- 更新 `pool_metrics_history.json`
- 输出 `pool_ic_raw.png` 与 `pool_ic_abs.png`

## 9. 策略梯度更新

当前 loss：

1. `adv = (reward - mean) / (std + 1e-5)`
2. `loss = mean_t(-log_prob_t * adv)`（按时间步累加）
3. `AdamW` 反向更新
4. 若启用 LoRD，额外做低秩衰减步

说明：这是 REINFORCE 风格更新，value head 目前未参与基线损失。

## 10. Checkpoint、Best Pool 与 Resume

### 10.1 保存内容

每次 `save_checkpoint(step, tag)` 保存：

- `model_state_dict`
- `optimizer_state_dict`
- `step`
- `best_score`, `best_formula`
- `best_pool_score`（如果存在）
- `training_history`
- `alpha_pool_path`（相对路径，指向 `checkpoints/alpha_pool/{tag}.pkl`）

并支持：

- step checkpoint：`step_XXXXXX.pt` + 同步复制到 `latest.pt`
- best checkpoint：`best.pt` + `best_step_XXXXXX.pt`
- final checkpoint：`final.pt`

### 10.2 best pool 快照

当 pool 分数创新高时，额外写：

- `best_alpha_pool.pkl`
- `best_alpha_pool_meta.json`（step、score、pool_size 等）

### 10.3 Resume 语义

支持两类恢复：

1. 模型 checkpoint：`--resume-model-checkpoint`
2. 独立 pool 文件：`--resume-alpha-pool`

优先级：

- 若同时给了 checkpoint 和 pool 路径：
  - 模型/优化器从 checkpoint 恢复
  - pool 从显式 pool 路径恢复（覆盖 checkpoint 内的 pool）
- `start_step = loaded_step + 1`

## 11. 训练产物总览

一次 run（`runs/<timestamp>/`）通常包含：

- `config_snapshot.json`
- `best_meme_strategy.json`
- `training_history.json`
- `vocab.json`
- `checkpoints/*.pt`
- `checkpoints/alpha_pool/*.pkl`
- `best_alpha_pool.pkl`
- `best_alpha_pool_meta.json`
- `pool_artifacts/*`
- `pool_features/*`
- `pool_features_removed/*`
- `pool_ic_raw.png`, `pool_ic_abs.png`

## 12. 端到端伪代码

```text
load config -> load data -> build model/vocab -> optionally resume
for step in [start_step, train_steps):
    init rewards = NO_EOS_PENALTY
    start process pool executor
    init per-sample formula buffer + type stack
    for t in [0, max_formula_len):
        logits = model(prefix)
        mask illegal tokens by stack/type/finishability/max_stack
        sample action
        if action == EOS:
            submit formula for async eval
            mark sample finished
        else:
            append token and update type stack
        stop if all finished
    collect eval futures
    compute rewards (pool-mode or direct backtest mode)
    policy gradient update
    log + periodic checkpoint
save final checkpoint + summary artifacts
```

---

如果你后续还会继续改训练目标（比如加入 value loss、PPO clipping、length-aware sampling），建议直接在本文补一节“训练目标版本历史”，避免团队成员对 reward 定义产生歧义。
