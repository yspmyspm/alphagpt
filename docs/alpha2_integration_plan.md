# Alpha2 集成开发文档

## 1. 目标

本文档回答两个问题：

1. 当前 `main/engine.py` 里，哪些职责应该下沉到基座类。
2. 如何把论文 **Alpha 2: Discovering Logical Formulaic Alphas using Deep Reinforcement Learning** 的方法，落到当前框架中。

注意：这里先做架构设计和开发计划，不直接实现 Alpha2 代码。

## 2. 先给结论

当前框架已经有三块比较稳定的基础设施：

- `ExprEval`：表达式执行与特征计算
- `AlphaPool`：单因子评价、相关性评价、组合池维护
- `main/utils/engine_base.py`：Ray 连接、checkpoint、RPN 合法性

真正不稳定、并且应该和算法解耦的，是 `main/engine.py` 里的这两部分：

- `REINFORCE` 训练循环
- “候选表达式 -> ExprEval -> AlphaPool -> reward/pool update”的异步评估流水线

如果后面要接 Alpha2，最重要的不是直接改现有 `AlphaEngine`，而是先把 `AlphaEngine` 拆成：

1. 通用运行时基座
2. 通用候选表达式评估基座
3. 表达式表示层基座（当前是 RPN，未来还会有 program/instruction）
4. 具体算法引擎（当前 REINFORCE；未来 Alpha2 的 MCTS + policy/value）

## 3. 当前框架与论文方法的差异

### 3.1 论文 Alpha2 的核心

论文的关键点不是“RL 发现 alpha”这么简单，而是四个组合：

- 用 **instruction/program** 而不是 token 级 RPN 作为搜索状态
- 用 **RL-guided MCTS** 而不是纯采样 + REINFORCE
- reward 采用 **增量 Perf**：`r_t = Perf(s_{t+1}) - Perf(s_t)`
- 扩展节点前做 **dimension consistency pruning**

论文里的 `Perf` 主要是：

- `Perf(alpha) = (1 - MaxCorr(alpha, G)) * IC(alpha, target)`

同时论文在 MCTS backup 上做了改造：

- `Q(s, a) = r(s, a) + beta * mean(V_s) + (1 - beta) * max(V_s)`
- 且 `V_s` 只保留 top-k values

### 3.2 你当前框架的现实情况

当前 `main/engine.py` 的范式是：

- 状态：RPN partial token sequence
- 动作：下一个 token
- 搜索：自回归采样
- 学习：REINFORCE
- reward：终局 reward，由 `AlphaPool` 返回的指标拼装
- 约束：RPN stack legality + 参数窗口合法性

当前框架已经具备的、可以直接复用的能力：

- `ExprEval` 的表达式执行能力
- `AlphaPool` 的单因子 IC / 多样性 / pool hypothetical metrics
- 异步 Ray evaluation pipeline
- 运行目录、checkpoint、resume

当前框架还没有的能力：

- instruction 级 program state
- register 语义
- 维度系统
- MCTS tree / node / backup
- policy/value 驱动的搜索闭环
- partial program 的增量 reward

## 4. engine 里哪些函数应该放到基座类

### 4.1 现有 `AlphaEngine` 中，建议下沉的函数

#### A. 应下沉到运行时基座 `AlphaEngineBase`

这些函数和具体算法基本无关：

- `_normalize_optional_path`
- `_json_compact_number_lists`
- `_write_capabilities_snapshot`

原因：

- 这些都是路径、artifact、序列化层面的运行时工具。
- 无论是当前 `AlphaEngine`，还是未来 `Alpha2Engine`，都会复用。

#### B. 应下沉到新的“候选表达式评估基座” `CandidateEvalEngineBase`

这一层建议新增，专门承载“候选表达式异步评估 + pool 交互”。

建议迁移：

- `_build_expr_eval_kwargs`
- `_start_formula_eval_job`
- `_promote_expr_jobs`
- `_collect_pool_eval_jobs`
- `_update_feature_leaderboard`
- `_print_factor_batch_summary`
- `_print_pool_update_summary`
- `_should_update_pool_from_eval_result`

这里面有一部分需要做成“模板方法 + hook”，不能原样硬塞进 base：

- `_reward_from_eval_result`
- `_feature_perf_score_from_eval_result`

原因：

- 评估流水线本身是通用的。
- 但 reward 定义、leaderboard 排序分数、pool 纳入规则，在不同算法里会不一样。
- Alpha2 需要的 reward 更接近论文的 `Perf` / `delta Perf`，不能直接复用现在的 weighted ICIR 逻辑。

所以更合理的方式是：

- base 负责异步调度与结果归并
- subclass 负责“如何从 eval result 计算 reward / rank / selection”

#### C. 应保留在当前具体引擎 `AlphaEngine`

这些强绑定当前 REINFORCE 训练方案：

- `__init__` 里模型与 optimizer 初始化
- `_policy_gradient_step`
- `_log_step`
- `train`

原因：

- 这些函数服务的是“当前 autoregressive sampling + REINFORCE”范式。
- Alpha2 会改成 MCTS + policy/value 更新，训练主循环完全不同。

#### D. 建议拆分但不一定下沉到最底层 base

- `_print_training_start_banner`
- `_print_training_completion_summary`
- `_format_token_line`

这几项不适合继续全部放在 `AlphaEngine`，但也不应该塞到最底层 `AlphaEngineBase`。

更合适的做法：

- 把通用 run/runtime banner 放到 `AlphaEngineBase`
- 把 token / RPN schema 的展示逻辑放到 `RPNBasedAlphaEngine`
- 具体算法额外字段（例如 policy gradient、MCTS 参数）由具体 engine 补充

### 4.2 推荐的新基类结构

建议把 `main/utils/engine_base.py` 扩成下面这条继承链：

```text
AlphaEngineBase
  └─ RayServiceAlphaEngineBase
      └─ CandidateEvalEngineBase
          ├─ RPNBasedAlphaEngine
          │   └─ AlphaEngine                 # 当前 REINFORCE 版本
          └─ ProgramBasedAlphaEngine
              └─ Alpha2Engine                # 未来 Alpha2 版本
```

各层职责建议如下：

### `AlphaEngineBase`

- run dir / artifacts
- checkpoint / resume
- pool manifest
- 通用序列化工具

### `RayServiceAlphaEngineBase`

- 连接 `ExprEval` / `AlphaPool`
- 拉取 capabilities / pool status / data range

### `CandidateEvalEngineBase`

- candidate -> ExprEval -> AlphaPool 的异步调度
- Ray job promotion / collect
- leaderboard / best pool artifact
- 抽象 hook：
  - `candidate_to_expr_payload`
  - `candidate_to_name`
  - `reward_from_eval_result`
  - `feature_perf_score_from_eval_result`
  - `should_update_pool_from_eval_result`

### `RPNBasedAlphaEngine`

- 当前 token schema
- RPN stack legality
- token catalog dump

### `ProgramBasedAlphaEngine`

- Alpha2 的 instruction schema
- register state
- dimension propagation
- program -> RPN compiler

## 5. 对 Alpha2 的落地建议：不要硬改当前 `AlphaEngine`

最不建议的做法是：

- 在现有 `AlphaEngine.train()` 里直接塞 MCTS
- 继续把 Alpha2 当成“另一个 reward 配置”

因为论文的核心变化不是 reward 配置，而是：

- 搜索状态变了
- 动作空间变了
- tree backup 变了
- 约束系统变了

所以推荐策略是：

- 保留当前 `AlphaEngine` 作为 `RPN + REINFORCE` 基线
- 平行新增 `Alpha2Engine`
- 只共享运行时、评估流水线、服务连接、artifact 管理

这样后面做实验对比、回退、A/B 配置都会干净很多。

## 6. Alpha2 在当前框架中的实现策略

### 6.1 总体思路

最稳妥的接法不是重写 `ExprEval`，而是：

1. 在 `main` 侧实现 Alpha2 的 **program/instruction IR**
2. 把 program state 编译成当前 `ExprEval` 能吃的 **RPN token 序列**
3. 继续复用 `ExprEval.evaluate_many_refs()` 和 `AlphaPool.evaluate_batch()`

这条路径的优点：

- `ExprEval` 不需要推倒重来
- `AlphaPool` 也不需要重写
- 搜索表示层可以贴近论文
- 执行层仍然复用当前稳定基础设施

### 6.2 建议新增的模块

建议新增一个 `main/alpha2/` 目录，至少包含：

- `instruction_schema.py`
  - 定义 instruction、operand、operator catalog
- `program_state.py`
  - 定义寄存器、placeholder、当前 program state
- `dimension_system.py`
  - 定义 feature dimension、operator dimension rule、register dimension propagation
- `compiler.py`
  - 把 program / register expression 编译成当前 `ExprEval` 所需的 RPN token list
- `environment.py`
  - `reset / legal_actions / step / is_terminal / export_candidate`
- `reward.py`
  - `Perf` 与 `delta Perf` 计算逻辑
- `mcts.py`
  - node、PUCT、selection、expansion、backup、top-k value stats
- `model.py`
  - policy/value 网络
- `engine.py`
  - Alpha2 训练与搜索主循环

## 7. Alpha2 在你当前框架里的关键设计决策

### 7.1 表达式表示层：必须新增 Program IR

论文的 state 不是 RPN token prefix，而是 alpha program。

所以推荐 state 至少包含：

- `instructions: list[Instruction]`
- `registers: list[ExprNode | None]`
- `register_dims: list[Dimension | None]`
- `current_output_reg: int`
- `is_terminal: bool`

每执行一条 instruction：

- 更新寄存器内容
- 更新寄存器维度
- 若 `Reg0` 非空，则可以导出“当前 alpha”

这点非常关键，因为论文 reward 是逐步增量 reward，不是只在完整表达式结束时打分。

### 7.2 先复用 ExprEval，但要加一层 compiler

当前 `ExprEval` 最擅长的是：

- 输入 RPN token sequence
- 输出 series / payload

所以 Alpha2 不需要直接改写执行器，只需要：

- 把 `Reg0` 对应的表达式树编译成 RPN
- 用当前 `ExprEval` 执行

建议把 compiler 放在 `main` 侧，而不是 `ExprEval` 侧。因为：

- program/instruction 是搜索表示，不是执行表示
- 执行层应该继续保持“只接受表达式”的职责边界

### 7.3 维度系统应放在搜索环境，不放在 ExprEval

论文最重要的工程创新之一，是 **扩展节点前** 做 dimension pruning。

因此维度系统必须存在于：

- `ProgramBasedAlphaEngine` 或 `alpha2/environment.py`

而不应该存在于：

- `ExprEval` 的运行期报错逻辑

因为如果放到 `ExprEval`，就退化成“生成以后再过滤”，这和论文目标相反。

### 7.4 `AlphaPool` 可以复用，但 reward 要分层

当前 `AlphaPool` 已经返回：

- `feature_metrics`
- `diversity_metrics.max_corr`
- `pool_metrics_if_added`

这足以支持两套 reward：

#### 第一套：论文一致版

用于 Alpha2 主 reward：

```text
Perf(alpha) = compliance * (1 - max_corr) * overall_ic
delta_reward = Perf(next_state) - Perf(current_state)
```

这里建议：

- `overall_ic` 取 `feature_metrics["overall_ic"]`
- `max_corr` 取 `diversity_metrics["max_corr"]`
- `compliance` 继续沿用现有质量门控

#### 第二套：当前工程增强版

作为可选实验配置保留：

- daily/monthly ICIR
- pool hypothetical gain
- weighted reward

这样可以同时支持：

- 论文复现
- 和当前系统目标保持一致的工程优化版本

### 7.5 当前 `AlphaGPT` 不应直接强行套 Alpha2

当前 `main/alphagpt.py` 的模型结构更像：

- 自回归 token policy
- 一个未被充分使用的 critic head

它可以复用 backbone 思想，但不要直接把它当 Alpha2 模型。

Alpha2 的模型至少要适配：

- program/instruction state 编码
- instruction 级 action prior
- value prediction

更稳妥的方式是：

- 复用 transformer backbone 的实现风格
- 重新定义输入序列和输出 head

## 8. 分阶段开发计划

### Phase 0：先做基座重构

目标：

- 不改变现有训练行为
- 先把通用评估流水线从 `main/engine.py` 拆出来

任务：

- 新增 `CandidateEvalEngineBase`
- 把异步 ExprEval / AlphaPool 调度逻辑下沉
- 把 reward / leaderboard / pool selection 做成 hook
- 保持当前 `AlphaEngine` 行为不变

验收：

- 现有 `AlphaEngine` 能零行为变化运行
- `tests/test_pipeline.py` 语义不变

### Phase 1：补齐 Alpha2 的表示层

目标：

- 先拥有 program/instruction state
- 先拥有 dimension pruning
- 暂时不接 RL/MCTS

任务：

- 定义 instruction schema
- 定义 register update 规则
- 定义 dimension rule system
- 实现 program -> RPN compiler
- 给出若干手工 program 的编译与执行单测

验收：

- 手写 program 能稳定编译成 RPN
- 编译结果在 `ExprEval` 可执行
- 非法维度组合能在 expansion 前被拒绝

### Phase 2：先做一个“无学习”的 Alpha2 环境闭环

目标：

- 验证 environment、reward、cache、pool integration 是否正确

任务：

- 实现 `environment.step()`
- 支持 partial state 导出当前 `Reg0` alpha
- 接入 `CandidateEvalEngineBase`
- 做随机搜索 / beam search / greedy search 原型
- 验证 `Perf` 与 `delta Perf`

验收：

- 随机或启发式搜索可以跑通
- 每个中间 state 都能得到一致的 `Perf`
- 评估缓存命中正常

### Phase 3：实现 MCTS

目标：

- 先把搜索算法跑起来，再接神经网络

任务：

- `MCTSNode`
- selection / expansion / simulation / backup
- PUCT
- 论文里的 top-k value backup
- `beta * mean + (1 - beta) * max` 聚合

验收：

- 给定固定 policy prior 时，MCTS 能稳定展开
- 维度 pruning 和 legal action masking 生效
- 同一个 state 不重复创建节点

### Phase 4：接 policy/value 网络

目标：

- 用模型指导 MCTS，而不是纯启发式搜索

任务：

- 设计 state encoder
- 设计 instruction prior head
- 设计 value head
- 定义 self-play / replay / target 生成流程
- 定义 loss：policy loss + value loss + regularization

验收：

- policy/value 训练可以收敛
- 搜索质量优于随机/beam baseline

### Phase 5：与当前 AlphaPool 训练目标对齐

目标：

- 让 Alpha2 在你的工程目标下可用，而不只是论文复现

任务：

- 增加 reward mode 配置
- 对比：
  - 论文 Perf
  - 当前 weighted reward
  - 是否纳入 `pool_metrics_if_added`
- 决定发现 alpha 后写入 pool 的节奏

验收：

- 可以跑论文一致配置
- 也可以跑工程增强配置
- 两种模式输出清晰可比较

## 9. 需要新增/调整的配置

建议在 `main/configs/config.json` 里增加 `alpha2` 段，而不是把参数塞进原有 `training`：

```json
{
  "alpha2": {
    "enabled": false,
    "engine_type": "alpha2",
    "program": {
      "num_registers": 2,
      "max_instructions": 15
    },
    "dimensions": {
      "feature_dims": {
        "open": "price",
        "close": "price",
        "high": "price",
        "low": "price",
        "vwap": "price",
        "volume": "volume"
      }
    },
    "mcts": {
      "num_simulations": 128,
      "puct_c_base": 1.25,
      "top_k_backup": 8,
      "beta_mean_max": 0.5
    },
    "reward": {
      "mode": "paper_perf",
      "use_delta_reward": true
    }
  }
}
```

这样做的原因：

- 避免把 Alpha2 参数污染当前 REINFORCE 配置
- 后面保留多引擎并行实验能力

## 10. 风险点

### 10.1 最大风险：动作空间爆炸

论文 action 是 instruction，不是 token。若直接把：

- operator
- operand1
- operand2
- operand3

笛卡尔积平铺成一个大 action id，动作空间会非常大。

建议：

- 环境仍然按 instruction step 运作
- 但模型 head 采用 factorized prediction：
  - 先 operator
  - 再 operand slots
- 最终在 MCTS expansion 阶段组装合法 instruction

### 10.2 第二个风险：partial state 评估成本高

论文 reward 依赖中间 state 的 `Perf`。

如果每条边都实时走：

- program -> RPN
- ExprEval
- AlphaPool

成本会很高。

必须提前设计缓存：

- key：program hash / compiled expression hash
- value：ExprEval payload、AlphaPool eval result、Perf

### 10.3 第三个风险：维度规则不是单纯 price/volume 二元分类

你当前 operator 集合比论文的演示复杂很多，特别是：

- `Divide`
- `LogRatio`
- `ts_corr`
- `ts_cov`
- `ts_beta_binary`
- `ts_reg_resid_binary`

这些操作需要明确的维度传播规则。

建议不要一开始就给所有 operator 补维度系统，而是：

1. 先选一个 Alpha2 operator 子集
2. 先把维度规则闭环跑通
3. 再逐步扩 operator coverage

## 11. 推荐的最小可行版本（MVP）

如果目标是先把 Alpha2 真正接入，而不是一步到位，建议 MVP 定义如下：

- 只支持 2 个寄存器
- 只支持论文里最核心的 unary / binary / ts unary / ts binary 子集
- 只支持 `open/close/high/low/vwap/volume`
- 先实现 dimension pruning
- 先实现 `Perf = compliance * (1 - max_corr) * overall_ic`
- 先做 MCTS + heuristic/value stub
- 最后再接训练网络

也就是说：

- **先把 Alpha2 的 environment 和 search 跑通**
- **再把 learning 接上**

这是风险最低的顺序。

## 12. 建议的验收顺序

1. 基座重构后，当前 `AlphaEngine` 行为不变。
2. 手写 Alpha2 program 能编译成 RPN，并在 `ExprEval` 执行成功。
3. 维度非法的 instruction 在扩展前被拦截。
4. partial state 能稳定计算 `Perf`。
5. MCTS 能在固定 prior 下稳定搜索。
6. policy/value 网络接入后，搜索结果优于随机搜索基线。

## 13. 最后给你的架构建议

如果你的目标真的是“把论文方法实现到现有框架”，那就应该：

- **复用执行层和评价层**
- **重做搜索表示层和搜索算法层**

具体来说：

- `ExprEval`、`AlphaPool` 尽量不动
- `main/engine.py` 不要继续膨胀
- 先把候选表达式评估流水线下沉为 base
- 以平行新引擎的方式实现 `Alpha2Engine`

这样以后你会同时拥有：

- 一个 `RPN + REINFORCE` 基线
- 一个 `Program + MCTS + policy/value` 的 Alpha2 版本

这才是后续做实验、做论文复现、做工程优化时最稳的结构。
