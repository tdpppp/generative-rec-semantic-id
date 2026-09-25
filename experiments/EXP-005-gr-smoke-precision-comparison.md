# EXP-005：GR Smoke Precision Comparison

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-005` |
| 阶段 / 状态 | Qwen2 GR / `completed` |
| 前置实验 | `EXP-004` |
| 对照变体 | FP16、BF16 |
| 基础模型 | Qwen2.5-0.5B |
| 数据规模 | 最多读取 500 条用户序列；实际训练 40 个样本、20 steps |
| 是否通过 smoke 门槛 | BF16 通过；FP16 数值稳定性未通过 |
| 是否允许进入下一阶段 | 允许使用 BF16 方案开展后续 GR 实验 |
| 核心结论 | 两种精度均跑通训练与保存链路；BF16 未出现非有限梯度，优于 FP16 |

两次运行只改变混合精度模式和输出目录，其他显式配置一致，因此合并为一个对照实验。

## 2. 目标与变化

目标是验证 Qwen2.5-0.5B 能否使用 `EXP-004` 生成的三层 Semantic ID 完成 GR 训练、日志记录和 checkpoint 保存，并比较 FP16 与 BF16 的数值稳定性。

| 变体 | 配置文件 | 精度 | 输出目录 |
|---|---|---|---|
| A | `gr/gr_train_qwen25_05b_smoke.json` | FP16 | `outputs/gr/checkpoints/qwen25_05b_smoke/` |
| B | `gr/gr_train_qwen25_05b_smoke_bf16.json` | BF16 | `outputs/gr/checkpoints/qwen25_05b_smoke_bf16/` |

本实验不验证生成质量、推荐效果或模型收敛，只验证小规模训练链路和数值稳定性。

## 3. 数据与配置

### 3.1 共同配置

| 项目 | 内容 |
|---|---|
| 基础模型 / tokenizer | `outputs/models/Qwen2.5-0.5B` |
| 用户序列 | `data/TencentGR_1M/seq`（配置中为 `../data/TencentGR_1M/seq`） |
| Semantic ID mapping | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` |
| mapping 商品数 | 386,364 |
| Semantic ID 深度 | 3 |
| Semantic ID 空间 | `[2048, 2048, 1024]` |
| max train samples | 500 |
| max sequence length | 100 |
| response flag | true |
| max steps / epochs | 20 / 1；`max_steps` 优先 |
| per-device batch size | 2 |
| gradient accumulation | 1 |
| learning rate | 1e-4 |
| weight decay | 0.01 |
| logging / save steps | 1 / 20 |
| dataloader workers | 2 |
| 随机种子 | 2025 |

根据训练状态，20 steps、batch size 2，共处理 40 个训练样本。两次运行均未配置验证集或生成评估。

### 3.2 唯一变量

| 配置 | FP16 变体 | BF16 变体 |
|---|---:|---:|
| `fp16` | true | false |
| `bf16` | false | true |

## 4. 执行与产物

| 项目 | FP16 | BF16 |
|---|---:|---:|
| 完成时间 | 2026-09-23 14:44 | 2026-09-23 14:53 |
| 训练运行时长 | 11.998 秒 | 12.468 秒 |
| 峰值显存 | 未记录 | 未记录 |
| 磁盘占用 | 约 7.5 GB | 约 7.5 GB |
| 是否出现训练崩溃 | 否 | 否 |
| 完成 steps | 20 / 20 | 20 / 20 |
| checkpoint | 已保存 | 已保存 |

每个输出目录同时包含最终模型和 `checkpoint-20`。单目录中约 2.0 GB 的最终模型、约 2.0 GB 的 checkpoint 模型和约 4.0 GB 的 optimizer state 是磁盘占用的主要来源。

| 变体 | 日志 | Trainer state | 最终模型 |
|---|---|---|---|
| FP16 | `outputs/gr/checkpoints/qwen25_05b_smoke/train.log` | `outputs/gr/checkpoints/qwen25_05b_smoke/checkpoint-20/trainer_state.json` | `outputs/gr/checkpoints/qwen25_05b_smoke/model.safetensors` |
| BF16 | `outputs/gr/checkpoints/qwen25_05b_smoke_bf16/train.log` | `outputs/gr/checkpoints/qwen25_05b_smoke_bf16/checkpoint-20/trainer_state.json` | `outputs/gr/checkpoints/qwen25_05b_smoke_bf16/model.safetensors` |

## 5. 结果

### 5.1 汇总对比

| 指标 | FP16 | BF16 |
|---|---:|---:|
| 首步 loss | 16.8068 | 16.8301 |
| 末步 loss | 12.4447 | 11.7445 |
| 最低单步 loss | 11.8694 | 11.6875 |
| 最高单步 loss | 22.3855 | 21.8744 |
| Trainer 汇总 train loss | 14.8445 | 13.1746 |
| 有限 grad norm | 11 / 20 | 20 / 20 |
| `NaN/Inf` grad norm | 9 / 20 | 0 / 20 |
| 最大有限 grad norm | 419.83 | 2,761.28 |
| checkpoint 保存 | 成功 | 成功 |

### 5.2 FP16 观察

FP16 完成了全部 20 steps，但第 1、2、3、4、6、8、9、18、20 步出现 `NaN` 或 `Inf` grad norm，共 9/20 步。loss 本身保持有限，最终 checkpoint 也成功保存，因此工程链路已跑通；但非有限梯度占比为 45%，不能视为稳定的训练配置。

部分 step 的学习率连续重复，和非有限梯度出现位置相邻。这与 AMP 梯度缩放在异常梯度时跳过 optimizer/scheduler step 的行为相符，但本实验没有单独记录 GradScaler 状态，因此只作为解释，不作为已证实结论。

### 5.3 BF16 观察

BF16 完成全部 20 steps，20 个 grad norm 均为有限值，没有出现 `NaN/Inf`。最高 grad norm 达到 2,761.28，仍存在较大的梯度波动，但没有导致数值溢出或训练中断。

BF16 的 Trainer 汇总 train loss 为 13.1746，低于 FP16 的 14.8445。不过本实验只有 20 steps，且两次运行即使使用相同 seed，也可能因混合精度数值路径不同而产生不同样本更新轨迹，因此该差异只能作为 smoke 观察，不能作为模型效果结论。

## 6. 结论与后续

两种精度都验证了以下工程链路：

- 加载 Qwen2.5-0.5B 和 tokenizer；
- 扩展 Semantic ID token；
- 加载 386,364 条 item-to-token mapping；
- 构造 GR 训练样本；
- 完成前向、反向和参数更新；
- 记录日志并保存 checkpoint、模型和 tokenizer。

FP16 虽然完成训练，但存在 9/20 个非有限 grad norm，未通过数值稳定性 smoke。BF16 没有非有限梯度，完成全部训练和保存流程，因此 BF16 是后续 GR 实验的默认精度方案。

后续实验应继续使用 BF16，并至少补充：

- 更长训练过程中的 loss 和 grad norm 稳定性；
- validation loss；
- Semantic ID 生成有效率；
- 生成结果能否映射回有效 item；
- HR、NDCG、Recall 等正式推荐指标。

在这些指标完成前，本实验只能证明 GR 工程链路和 BF16 数值稳定性，不能证明生成式推荐效果。
