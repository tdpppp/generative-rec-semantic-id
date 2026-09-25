# EXP-004：Balanced RQ-VAE

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-004` |
| 阶段 / 状态 | RQ-VAE / `completed` |
| 前置实验 | `EXP-003` |
| 数据规模 | 386,364 items |
| 是否通过质量门槛 | 是 |
| 是否允许进入下一阶段 | 允许进行 GR smoke |
| 核心结论 | 平衡训练缓解了 codebook collapse，推理后 collision rate 从 99.08% 降至 2.89% |

## 2. 目标与变化

目标是验证缩短 warmup、启用训练期 Sinkhorn 平衡并提高 collision 评估频率后，能否缓解 `EXP-003` 的 codebook collapse。

| 配置 | EXP-003 | EXP-004 |
|---|---|---|
| epochs | 20 | 100 |
| warmup | 50 epochs | 2 epochs |
| Sinkhorn | 未启用，`sk_epsilons=[0, 0, 0]` | 启用，`sk_epsilons=[0.01, 0.01, 0.01]` |
| Sinkhorn iterations | 未记录 | 20 |
| collision 评估 | 最终评估 | 每 10 epochs |
| 输出目录 | `outputs/10k/` | `outputs/10k_balanced/` |

本实验同时改变了训练轮数、warmup 和 Sinkhorn 配置，因此可以确认组合方案有效，但不能单独归因到某一个变量。

## 3. 数据与配置

| 项目 | 内容 |
|---|---|
| 输入产物 | `outputs/10k/emb/embeddings.npz` |
| 输入方式 | `outputs/10k_balanced/emb -> ../10k/emb` |
| 输入 shape | 386,364 x 32, float32 |
| item ID | 386,364 个，全部唯一 |
| embedding NaN/Inf | 全部有限 |
| epochs / batch size | 100 / 4,096 |
| learning rate / optimizer | 3e-4 / AdamW |
| scheduler / warmup | constant / 2 epochs |
| eval step | 10 epochs |
| codebook sizes | `[2048, 2048, 1024]` |
| Sinkhorn | `sk_epsilons=[0.01, 0.01, 0.01]`，`sk_iters=20` |
| K-means | 启用，100 iterations |
| embedding dimension | 32 |
| loss | MSE，quantization weight 1.0，beta 0.25 |
| encoder layers | `[4096, 2048, 1024, 512, 256, 128, 64]` |
| 随机种子 | 2025 |
| 输出目录 | `outputs/10k_balanced/` |

原始 shell 命令未单独保存；以上有效配置来自 checkpoint 中保存的训练参数。

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 训练日期 | 2026-09-23 |
| 训练时长 | 约 13 分 14 秒，根据 TensorBoard 与训练日志时间计算 |
| 峰值显存 | 未记录 |
| 磁盘占用 | 3.1 GB |
| 是否出现训练崩溃 | 否，100 epochs 完整结束 |

| 产物 | 路径 | 用途 | 状态 |
|---|---|---|---|
| train log | `outputs/10k_balanced/rqvae/train.log` | 训练过程和最终最佳指标 | 可用 |
| TensorBoard | `outputs/10k_balanced/rqvae/tensorboard/` | loss 和 collision 曲线 | 可用 |
| best collision checkpoint | `outputs/10k_balanced/rqvae/checkpoints/09-23-2026/best_collision_model.pth` | Semantic ID 推理 | 可用 |
| best loss checkpoint | `outputs/10k_balanced/rqvae/checkpoints/09-23-2026/best_loss_model.pth` | loss 最优对照 | 可用 |
| Semantic ID mapping | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` | GR 输入映射 | 可用于 GR smoke |
| summary | `outputs/10k_balanced/experiment_summary.json` | 自动汇总快照 | 可用 |

## 5. 结果

### 5.1 训练结果

| 指标 | 首次 | 最终 | 最佳 |
|---|---:|---:|---:|
| step total loss | 0.036816 | 0.029398 | 0.008183 |
| step reconstruction loss | 0.036815 | 0.002977 | 0.002291 |
| epoch train loss sum | 54.433960 | 2.304343 | 0.852173 |
| epoch reconstruction loss sum | 2.817891 | 0.247539 | 0.242756 |
| collision rate | 75.49%（epoch 9） | 8.55%（epoch 99） | 6.21%（epoch 49） |

collision rate 每 10 epochs 的记录：

| Epoch | Collision rate |
|---:|---:|
| 9 | 75.49% |
| 19 | 25.81% |
| 29 | 10.49% |
| 39 | 7.07% |
| 49 | 6.21% |
| 59 | 6.49% |
| 69 | 6.32% |
| 79 | 7.28% |
| 89 | 8.74% |
| 99 | 8.55% |

最佳 collision 出现在 epoch 49，之后没有继续改善，末期略有回升。后续推理应优先使用 `best_collision_model.pth`，而不是最终 epoch checkpoint。

### 5.2 Semantic ID 推理结果

| 指标 | 结果 |
|---|---:|
| 输入商品 | 386,364 |
| 唯一 item ID | 386,364 |
| item 覆盖率 | 100% |
| 唯一 Semantic ID | 375,197 |
| 碰撞商品 | 11,167 |
| 推理后 collision rate | 2.89% |
| 第一层 code 使用 | 1,130 / 2,048（55.18%） |
| 第二层 code 使用 | 840 / 2,048（41.02%） |
| 第三层 code 使用 | 433 / 1,024（42.29%） |

与 `EXP-003` 对比：

| 指标 | EXP-003 | EXP-004 |
|---|---:|---:|
| 推理后 collision rate | 99.08% | 2.89% |
| 唯一 Semantic ID | 3,552 | 375,197 |
| 第一层 code 使用 | 12 / 2,048 | 1,130 / 2,048 |
| 第二层 code 使用 | 20 / 2,048 | 840 / 2,048 |
| 第三层 code 使用 | 235 / 1,024 | 433 / 1,024 |

## 6. 结论与后续

平衡训练组合显著缓解了 codebook collapse。推理后 collision rate 为 2.89%，低于进入 GR 的当前门槛 10%，item 覆盖率和唯一率均为 100%，各层 codebook 使用也不再局限于个位数或十位数 code。因此本实验通过当前 RQ-VAE 质量门槛，生成的 mapping 可以用于 GR smoke。

限制：

- 本实验一次改变了多个变量，不能判断改善主要来自训练轮数、warmup 还是 Sinkhorn；
- 训练期最佳 collision 出现在 epoch 49，之后略有回升，应保留并使用最佳 checkpoint；
- 通过当前工程门槛不等于已经验证最终生成式推荐效果。

下一步使用 `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` 进行 GR smoke，并在 GR 实验中记录数据构造、训练稳定性、生成有效率和推荐指标。

