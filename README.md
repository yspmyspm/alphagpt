# AlphaGPT

基于强化学习的可解释 alpha 表达式生成系统。  
核心流程是：`生成表达式 -> 执行成因子 -> 回测打分 -> 更新生成模型`，并可维护一个持续优化的 `alpha pool`。

详细算法文档见 [docs/algorithm_pipeline.md](docs/algorithm_pipeline.md)。

## 特性

- RPN 表达式生成，带严格约束解码（类型栈合法性 + 可收敛约束 + 最大栈深约束）。
- EOS 即时异步评估：短表达式先结束先评测，不阻塞后续样本。
- IC/ICIR + 合规性（finite/distribution/halflife）联合奖励。
- Alpha pool 联合更新：候选去冗余、组合权重优化、容量裁剪。
- 完整 checkpoint 与 resume：
  - 可从模型 checkpoint 恢复；
  - 可单独指定 alpha pool 快照覆盖恢复。

## 目录结构

```text
alphagpt/
├─ engine.py                     # 训练主入口（rollout + 评估 + PG 更新）
├─ utils/engine_base.py          # 保存/加载/checkpoint 等基础能力
├─ alphagpt.py                   # 模型定义（LoopedTransformer + MTPHead）
├─ vm.py                         # RPN Stack VM 执行器
├─ backtest.py                   # 因子评估入口
├─ data_loader.py                # Feather 数据加载
├─ config.json                   # 主配置文件
├─ config.py                     # 兼容层（转发到 configs/config.py）
├─ configs/                      # 配置模块 + 可选子配置
├─ helpers/
│  ├─ eval_worker.py             # 多进程评估 worker
│  ├─ reward_metrics.py          # IC/ICIR/compliance 计算
│  └─ myops.py                   # 操作符注册
└─ alphapool/
   ├─ pool_state.py              # pool 状态与持久化
   ├─ batch_evaluator.py         # batch reward + joint pool update
   ├─ ensemble_trainer.py        # 组合权重优化（L-BFGS-B）
   └─ pool_monitor.py            # pool 指标与产物监控
```

## 环境要求

- Python 3.10+（推荐 3.11）
- Windows / Linux 均可（Windows 下使用 `spawn` 多进程启动）

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

默认读取：

- 特征：`data/olhcv.feather`
- 收益：`returns/30min.feather`

要求：

- 文件为 Feather 格式；
- 若有 `_time` 列，会作为时间索引；
- `returns` 文件需包含 `configs/config.json` 中 `returns_column` 指定的列（默认 `returns`）。

## 快速开始

1. 修改 `config.json`（至少确认数据路径和训练参数）。
2. 直接训练：

```bash
python engine.py
```

3. 指定配置文件路径（可选）：

```bash
python engine.py --config config.json
```

4. 指定恢复模型 checkpoint：

```bash
python engine.py --resume-model-checkpoint runs/20260330_120000/checkpoints/latest.pt
```

5. 同时恢复模型 + 指定 pool（pool 会覆盖 checkpoint 内嵌 pool）：

```bash
python engine.py --resume-model-checkpoint runs/20260330_120000/checkpoints/latest.pt --resume-alpha-pool runs/20260330_120000/best_alpha_pool.pkl
```

## 配置说明（`config.json`）

主要分组：

- `runtime`
  - `device`: `auto/cpu/cuda`
  - `eval_num_workers`: 评估进程数（`<=0` 时自动按 CPU 推断）
  - `logging_dir`: run 输出目录
  - `mp_start_method`: 多进程启动方式（默认 `spawn`）
- `data`
  - `feather_path`, `returns_dir`, `returns_filename`, `returns_column`, `feature_columns`
- `training`
  - `batch_size`, `train_steps`, `save_checkpoint_every`, `optimizer_lr`
- `resume`
  - `resume_model_checkpoint`, `resume_alpha_pool_path`
- `model`
  - `max_formula_len`, `gen_temperature`, `ts_parameters`, `max_decode_stack_size`
- `model_architecture`
  - `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `num_loops`, `dropout`, `mtp_num_tasks`
- `compliance`
  - `distribution_check_skew_limit`, `distribution_check_kurt_limit`
  - `score_finite_min_ratio`, `score_distribution_skew_limit`, `score_distribution_kurt_limit`
  - `score_halflife_min_corr`, `halflife_lag`
- `regularization`
  - `use_lord_regularization`, `lord_decay_rate`, `lord_num_iterations`
  - `lord_decay_keywords`, `lord_rank_monitor_keywords`, `lord_rank_log_every`
- `scoring`
  - `no_eos_penalty`, `execute_fail_penalty`, `missing_high_penalty`, `low_std_penalty_base`, `low_std_threshold`, `compliance_fail_penalty`
  - `use_smooth_reward`, `backtest_penalty`, `icir_missing_gamma`, `icir_missing_eps`
  - `reward_weight_daily_icir`, `reward_weight_monthly_icir`, `reward_weight_ic_score`
- `alpha_pool`
  - `use_alpha_pool`, `alpha_pool_size`, `alpha_missing_threshold`, `alpha_max_corr_min_points`, `ensemble_maxiter`

数值稳定性常量（如 `policy_advantage_eps`、`icir_std_eps` 等）已硬编码在 `ModelConfig` 类中，不再通过配置文件暴露。

配置文件支持 `sub_config_paths` 字段，用于引用子配置文件（按路径深度合并）。
例如可将 `model_architecture` 拆入 `configs/model_architecture.json`，在主配置中引用：

```json
{
  "sub_config_paths": {
    "arch": "configs/model_architecture.json"
  }
}
```

也可通过环境变量切换配置文件：

```bash
set CONFIG_PATH=C:\path\to\your_config.json
python engine.py
```

## 训练产物

每次训练会生成 `runs/<timestamp>/`，常见文件：

- `config_snapshot.json`: 本次实际生效配置快照
- `training_history.json`: step 级训练历史
- `best_meme_strategy.json`: 当前 best 公式摘要
- `vocab.json`: token 词表与 token id
- `checkpoints/`
  - `latest.pt`, `final.pt`, `best.pt`, `step_XXXXXX.pt`, `best_step_XXXXXX.pt`
  - `alpha_pool/<tag>.pkl`（随 checkpoint 的 pool 快照）
- `best_alpha_pool.pkl` + `best_alpha_pool_meta.json`
- `pool_artifacts/`：组合预测与 pool 指标历史
- `pool_features/`、`pool_features_removed/`：pool 入池/出池因子明细
- `pool_ic_raw.png`、`pool_ic_abs.png`：pool 指标曲线

## 训练流程摘要

1. 加载特征与收益，构建词表与模型。
2. 约束解码生成表达式 token 序列。
3. 任意样本遇到 EOS 即提交异步评测。
4. 计算奖励并（可选）联合更新 alpha pool。
5. 用策略梯度更新模型。
6. 持续记录指标，按步保存 checkpoint。

详细过程和公式见 [docs/algorithm_pipeline.md](docs/algorithm_pipeline.md)。

## 常见问题

### 1. 为什么很多样本拿到 `no_eos_penalty`？

通常是 `max_formula_len` 太小、操作符空间太大或约束太紧。优先检查：

- `model.max_formula_len`
- `model.max_decode_stack_size`
- 操作符和参数规模（`helpers/myops.py`、`ts_parameters`）

### 2. 如何只恢复模型不恢复 pool？

只传 `--resume-model-checkpoint` 即可。  
若还传了 `--resume-alpha-pool`，则会用该 pool 覆盖 checkpoint 内的 pool。

### 3. Windows 多进程报错怎么办？

从命令行直接运行 `python engine.py`。  
不要在不支持 `spawn` 的交互环境里直接复用训练入口。
