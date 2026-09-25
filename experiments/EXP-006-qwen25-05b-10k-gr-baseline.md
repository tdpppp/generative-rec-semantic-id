# EXP-006：Qwen2.5-0.5B 10k GR Baseline

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-006` |
| 阶段 / 状态 | Qwen2 GR / `completed` |
| 前置实验 | `EXP-004`、`EXP-005` |
| 基础模型 | Qwen2.5-0.5B Base |
| 数据规模 | 10,000 users；875,365 train / 10,000 validation / 10,000 test samples |
| 是否通过训练质量检查 | 是 |
| 是否完成正式推荐评估 | 已完成 Semantic-ID 级 Top-10 测试；item 级碰撞消歧尚未完成 |
| 核心结论 | 训练稳定且受约束生成 100% 合法，但 HR@10 仅 0.81%，目录覆盖率仅 0.338%，推荐效果与多样性仍不足 |

## 2. 目标与实验过程

目标是使用 `EXP-004` 的三层 Semantic ID mapping，在 10k 用户 profile 上训练 Qwen2.5-0.5B GR 基线，验证完整训练、时序验证和最佳 checkpoint 保存流程。

训练分为两段完成：第一段运行到 step 20,000 后停止，第二段使用 `gr/gr_train_qwen25_05b_10k_v1_resume.json` 从 `checkpoint-20000` 恢复，并完成到 step 27,356。恢复过程中日志报告 `lm_head.weight` 缺失；该模型使用 tied input/output embeddings，训练与最终模型保存均正常完成，但此警告仍保留在实验限制中。

训练完成后，使用根输出目录中保存的最佳模型，对 10k test split 运行受约束的 Top-10 生成评估。解码由 Semantic ID 前缀树约束，只允许生成 mapping 中存在的三层组合。

## 3. 数据与配置

### 3.1 数据

| 项目 | 内容 |
|---|---|
| profile | `data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82` |
| 用户数 | 10,000 |
| 训练样本 | 875,365 |
| 验证样本 | 10,000 |
| 测试样本 | 10,000 |
| 时序切分 | 训练使用倒数两个商品之前的滑动前缀；验证预测倒数第二个商品；测试预测最后一个商品 |
| Semantic ID mapping | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` |
| mapping 商品数 | 386,364 |
| Semantic ID 深度 / 空间 | 3 / `[2048, 2048, 1024]` |
| max sequence length | 100 |
| response flag | true，仅对目标 Semantic ID 计算 loss |

### 3.2 模型与训练配置

| 项目 | 内容 |
|---|---|
| 模型 / tokenizer | `outputs/models/Qwen2.5-0.5B` |
| precision | BF16 |
| epochs / total steps | 1 / 27,356 |
| train / eval batch size | 32 / 32 |
| gradient accumulation | 1 |
| learning rate | 3e-5 |
| scheduler / warmup | cosine / 3% |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| logging / eval / save steps | 20 / 2,000 / 2,000 |
| best model metric | validation loss，越低越好 |
| save limit | 2 checkpoints |
| dataloader workers | 4 |
| 随机种子 | 2025 |
| 初始配置 | `gr/gr_train_qwen25_05b_10k_v1.json` |
| 续训配置 | `gr/gr_train_qwen25_05b_10k_v1_resume.json` |
| 输出目录 | `outputs/gr/checkpoints/qwen25_05b_10k_v1/` |

### 3.3 测试配置

| 项目 | 内容 |
|---|---|
| 评估模型 | `outputs/gr/checkpoints/qwen25_05b_10k_v1/`，对应 step 26,000 最佳权重 |
| split | test |
| 测试用户 | 10,000 |
| top-k | 10 |
| 解码约束 | Semantic ID 前缀树；顺序固定为 `a -> b -> c`，组合必须存在于 mapping |
| 评估层级 | Semantic-ID 级 |
| mapping | `outputs/10k_balanced/emb_infer/sinkhorn/` |

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 完成日期 | 2026-09-23 |
| 运行方式 | step 0-20,000 初始训练；step 20,000-27,356 checkpoint 恢复训练 |
| 总活跃运行时间 | 约 4 小时；由两段日志时间估算 |
| 续训段 Trainer runtime | 3,642.85 秒（约 1 小时 43 秒） |
| 峰值显存 | 未记录 |
| checkpoint 与最终模型磁盘占用 | 约 14 GB |
| TensorBoard 磁盘占用 | 约 235 KB |
| 是否出现训练崩溃 | 训练曾中断，原因未记录；没有数值崩溃，且从 checkpoint 正常恢复 |
| 是否存在非有限 loss / grad norm | 否；1,367 条日志记录全部有限 |

| 产物 | 路径 | 状态 |
|---|---|---|
| 最终模型 | `outputs/gr/checkpoints/qwen25_05b_10k_v1/model.safetensors` | 已保存；内容为训练结束时加载的最佳模型 |
| 最佳 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_v1/checkpoint-26000/` | validation loss 5.4284 |
| 最终 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_v1/checkpoint-27356/` | 完整训练状态 |
| 训练日志 | `outputs/gr/checkpoints/qwen25_05b_10k_v1/train.log` | 可用 |
| Trainer state | `outputs/gr/checkpoints/qwen25_05b_10k_v1/checkpoint-27356/trainer_state.json` | 可用 |
| TensorBoard | `outputs/gr/tensorboard/qwen25_05b_10k_v1/` | 可用 |
| 测试指标 | `outputs/gr/checkpoints/qwen25_05b_10k_v1/test_metrics.json` | 已生成 |
| 测试预测 | `outputs/gr/checkpoints/qwen25_05b_10k_v1/test_predictions.jsonl` | 10,000 行，约 3.8 MB |
| 评估日志 | `outputs/gr/checkpoints/qwen25_05b_10k_v1/evaluate_test.log` | 可用 |

## 5. 结果

### 5.1 训练稳定性

训练日志每 20 steps 记录一次，共 1,367 条：

| 指标 | 结果 |
|---|---:|
| 首条 train loss | 16.3692 |
| 末条 train loss | 5.4421 |
| 最低记录 loss | 5.2851 |
| 最高记录 loss | 16.3692 |
| 记录 loss 均值 | 5.8432 |
| 首条 grad norm | 147.45 |
| 末条 grad norm | 22.86 |
| 最低 grad norm | 10.17 |
| 最高 grad norm | 492.80 |
| 非有限 loss | 0 / 1,367 |
| 非有限 grad norm | 0 / 1,367 |

loss 在训练早期快速下降，后期主要在约 5.3-5.5 范围内波动。梯度范数始终有限，说明 `EXP-005` 选择 BF16 后，完整规模训练没有复现 FP16 smoke 的 `NaN/Inf` 梯度问题。

Trainer 结束时打印 `train_loss=1.4597`，但该值不能作为完整 1 epoch 的平均 loss：本实验从 step 20,000 恢复后，Trainer 只累计续训段 loss，却使用全局 27,356 steps 计算汇总值。本文以完整 `trainer_state.json` 中的分段日志统计为准。

### 5.2 验证曲线

| Step | Epoch | Validation loss |
|---:|---:|---:|
| 2,000 | 0.07 | 6.3876 |
| 4,000 | 0.15 | 6.2661 |
| 6,000 | 0.22 | 6.0861 |
| 8,000 | 0.29 | 5.9461 |
| 10,000 | 0.37 | 5.8728 |
| 12,000 | 0.44 | 5.7947 |
| 14,000 | 0.51 | 5.7135 |
| 16,000 | 0.58 | 5.6387 |
| 18,000 | 0.66 | 5.5764 |
| 20,000 | 0.73 | 5.5220 |
| 22,000 | 0.80 | 5.4714 |
| 24,000 | 0.88 | 5.4394 |
| 26,000 | 0.95 | 5.4284 |

validation loss 从 6.3876 持续下降到 5.4284，13 次评估中没有出现回升。最佳 checkpoint 为 step 26,000；训练结束时 `load_best_model_at_end` 已加载该 checkpoint，并将其保存为根输出目录中的最终模型。

### 5.3 测试集生成与推荐指标

| 指标 | 结果 |
|---|---:|
| 测试用户 | 10,000 |
| Top-10 推荐位置 | 100,000 |
| 命中用户 | 81 |
| HR@10 | 0.0081（0.81%） |
| Recall@10 | 0.0081（0.81%） |
| NDCG@10 | 0.005499 |
| MRR@10 | 0.004696 |
| 合法格式率 | 100% |
| 合法 mapping 率 | 100% |
| 唯一推荐 Semantic ID | 1,270 |
| Semantic ID 目录覆盖率 | 0.003385（0.338%） |
| 歧义目标比例 | 0.0433（4.33%） |
| 不足 10 个预测的用户 | 0 |
| 单用户列表内重复 | 0 |

81 次命中的 rank 分布为：rank 1 命中 33 次、rank 2 命中 15 次，其余 33 次分布在 rank 3-10。由 predictions 重新计算得到的 HR、MRR 和 NDCG 与 `test_metrics.json` 完全一致。

合法格式率和 mapping 有效率均为 100%，证明受约束解码正确工作，所有用户都获得了 10 个互不重复、可映射的 Semantic ID。这解决了“能否生成合法候选”的工程问题，但没有解决推荐质量问题：HR@10 只有 0.81%，且 100,000 个推荐位置只涉及 1,270 个唯一 Semantic ID，目录覆盖率仅 0.338%，模型输出明显集中在很小的候选集合上。

4.33% 的测试目标对应存在碰撞的 Semantic ID，高于 mapping 的全局商品碰撞率 2.89%，说明碰撞在测试目标分布上的影响更明显。当前排名指标在 Semantic-ID 层级计算；命中一个有碰撞的 Semantic ID 不等于唯一命中目标 item，因此不能将这些结果直接表述为无歧义的 item 级 HR/NDCG/Recall。

## 6. 结论与后续

Qwen2.5-0.5B 10k GR 基线完成了训练、验证和受约束的测试集 Top-10 评估。checkpoint 恢复有效，BF16 数值稳定，validation loss 持续下降；受约束生成也达到 100% 格式合法率和 100% mapping 有效率。因此完整 GR 工程链路已经验证。

模型效果尚不理想。Semantic-ID 级 HR@10 只有 0.81%，NDCG@10 为 0.005499，目录覆盖率只有 0.338%，说明模型相关性和推荐多样性都较弱。后续重点不再是修复生成合法性，而应分析候选集中问题，比较更长训练、模型容量、解码策略和数据构造；同时需要增加 Semantic ID 碰撞后的 item 级消歧或重排，才能得到严格的 item 级推荐指标。
