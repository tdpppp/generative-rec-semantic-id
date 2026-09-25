# EXP-008：Qwen2.5-0.5B 10k 两轮训练

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-008` |
| 阶段 / 状态 | Qwen2 GR / `completed` |
| 对照实验 | `EXP-006`、`EXP-007` |
| 基础模型 | Qwen2.5-0.5B Base |
| 数据规模 | 10,000 users；875,365 train / 10,000 validation / 10,000 test samples |
| 主要变化 | 从预训练模型重新训练 2 epochs；其余显式超参数与 `EXP-006` 相同 |
| 是否通过训练与产物检查 | 是；训练无非有限 loss/grad norm，validation/test 产物检查均通过 |
| 核心结论 | 测试 HR@10 从 0.81% 提升至 1.13%，目录覆盖率从 0.338% 提升至 0.599%；延长到两轮有效，但绝对效果和覆盖率仍低 |

## 2. 目标与实验过程

本实验检验在相同模型、数据、batch size 和峰值学习率下，将训练从 1 epoch 延长到 2 epochs 是否能够改善推荐相关性和候选覆盖率。训练从 Qwen2.5-0.5B 预训练权重重新开始，并非从 `EXP-006` 的 checkpoint 续训，共完成 54,712 steps。

相对于 `EXP-006`，显式配置除 `num_train_epochs` 从 1 改为 2、输出目录变化外均保持一致。需要注意，训练使用 cosine scheduler 和 3% warmup，调度周期由总训练步数决定；epoch 数翻倍也同时延长了 warmup 和 cosine 调度周期。因此本实验验证的是完整的“两轮训练方案”，不能严格解释为仅追加一轮训练的单变量对照。

训练结束后加载 validation loss 最低的 step 54,000 权重，并分别在 validation 和 test split 上进行受约束 Top-10 生成评估。根输出目录的 `model.safetensors` 与 `checkpoint-54000/model.safetensors` SHA-256 一致，确认评估所用根模型是最佳 checkpoint，而不是最后的 step 54,712 权重。

## 3. 数据与配置

### 3.1 数据与评估口径

| 项目 | 内容 |
|---|---|
| profile | `data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82` |
| 用户数 | 10,000 |
| 训练 / 验证 / 测试样本 | 875,365 / 10,000 / 10,000 |
| 时序切分 | 训练使用倒数两个商品之前的滑动前缀；验证预测倒数第二个商品；测试预测最后一个商品 |
| Semantic ID mapping | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` |
| mapping 商品数 / 唯一 SID 数 | 386,364 / 375,197 |
| Semantic ID 深度 / 空间 | 3 / `[2048, 2048, 1024]` |
| max sequence length | 100 |
| response flag | true，仅对目标 Semantic ID 计算 loss |
| 评估层级 / Top-K | Semantic-ID 级 / 10 |
| 解码约束 | Semantic ID 前缀树；只允许生成 mapping 中存在的三层组合 |

### 3.2 模型与训练配置

| 项目 | 内容 |
|---|---|
| 模型 / tokenizer | `outputs/models/Qwen2.5-0.5B` |
| 初始化方式 | 从预训练模型重新开始，未恢复 checkpoint |
| precision | BF16 |
| epochs / total steps | 2 / 54,712 |
| train / eval batch size | 32 / 32 |
| gradient accumulation | 1 |
| learning rate | 3e-5 |
| scheduler / warmup | cosine / 3% |
| weight decay / max grad norm | 0.01 / 1.0 |
| logging / eval / save steps | 20 / 2,000 / 2,000 |
| best model metric | validation loss，越低越好 |
| save limit | 2 checkpoints |
| dataloader workers | 4 |
| 随机种子 | 2025 |
| 配置 | `gr/gr_train_qwen25_05b_10k_ep2_lr3e5_b32.json` |
| 输出目录 | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/` |

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 开始 / 训练及 validation 结束 | 2026-09-23 23:21 / 2026-09-24 07:00（UTC+8） |
| Trainer runtime | 27,191.04 秒，约 7 小时 33 分 |
| 含训练后 validation 评估总时长 | 约 7 小时 40 分 |
| test 评估时长 | 未记录 |
| 峰值显存 | 未记录 |
| checkpoint 与最终模型磁盘占用 | 约 14 GB |
| TensorBoard 磁盘占用 | 约 586 KB |
| 是否出现训练崩溃 | 否；进程 exit code 为 0 |

| 产物 | 路径 | 状态 |
|---|---|---|
| 最佳模型 | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/model.safetensors` | step 54,000 最佳权重 |
| 最佳 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/checkpoint-54000/` | validation loss 4.9557 |
| 最终 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/checkpoint-54712/` | 完整训练状态 |
| 训练日志 | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/train.log` | 可用 |
| TensorBoard | `outputs/gr/tensorboard/qwen25_05b_10k_ep2_lr3e5_b32/` | 可用 |
| validation 指标 / 预测 | `validation_metrics.json` / `validation_predictions.jsonl` | 10,000 用户，产物检查通过 |
| test 指标 / 预测 | `test_metrics.json` / `test_predictions.jsonl` | 10,000 用户，产物检查通过 |
| 运行状态 | `experiment_status.json` | `completed`，exit code 0 |

## 5. 结果

### 5.1 训练稳定性与验证曲线

训练日志每 20 steps 记录一次，共 2,735 条：

| 指标 | 结果 |
|---|---:|
| 首条 / 末条 train loss | 16.4755 / 4.8643 |
| 最低 / 平均记录 loss | 4.7008 / 5.4670 |
| 首条 / 末条 grad norm | 197.88 / 25.49 |
| 最低 / 最高 grad norm | 7.59 / 197.88 |
| 非有限 loss | 0 / 2,735 |
| 非有限 grad norm | 0 / 2,735 |
| Trainer 汇总 train loss | 5.4669 |

共执行 27 次 validation loss 评估，loss 从 step 2,000 的 6.4106 持续下降至 step 54,000 的 4.9557，没有出现回升。关键节点如下：

| Step | Epoch | Validation loss |
|---:|---:|---:|
| 2,000 | 0.07 | 6.4106 |
| 14,000 | 0.51 | 5.8020 |
| 26,000 | 0.95 | 5.4594 |
| 28,000 | 1.02 | 5.3974 |
| 40,000 | 1.46 | 5.0765 |
| 48,000 | 1.75 | 4.9709 |
| 52,000 | 1.90 | 4.9571 |
| 54,000 | 1.97 | **4.9557** |

最佳 validation loss 相比 `EXP-006` 的 5.4284 下降 0.4727，降幅约 8.71%。后半程仍在改善，但 step 52,000 到 54,000 只下降 0.0013，已接近平台区间。

### 5.2 Validation 与测试结果

| 指标 | Validation | Test |
|---|---:|---:|
| 用户数 | 10,000 | 10,000 |
| 命中用户 | 125 | 113 |
| HR@10 / Recall@10 | 0.0125（1.25%） | 0.0113（1.13%） |
| NDCG@10 | 0.007394 | 0.006899 |
| MRR@10 | 0.005841 | 0.005565 |
| 合法格式率 | 100% | 100% |
| 合法 mapping 率 | 100% | 100% |
| 唯一推荐 Semantic ID | 2,249 | 2,248 |
| Semantic ID 目录覆盖率 | 0.005994（0.599%） | 0.005992（0.599%） |
| 歧义目标比例 | 4.23% | 4.33% |

validation 和 test 产物均包含 10,000 行、10,000 个唯一用户，每位用户恰有 10 个候选；空候选、列表内重复和非法候选均为 0，重新统计的命中数与指标文件一致。测试集 113 次命中的排名分布为：rank 1 有 37 次，rank 2 有 11 次，rank 3 有 13 次，其余 52 次分布在 rank 4-10。

### 5.3 与既有实验和基线对比

| 指标 | Random (`EXP-007`) | MostPopular (`EXP-007`) | 1 epoch (`EXP-006`) | 2 epochs（本实验） |
|---|---:|---:|---:|---:|
| 命中用户 | 1 | 73 | 81 | 113 |
| HR@10 | 0.01% | 0.73% | 0.81% | **1.13%** |
| NDCG@10 | 0.000036 | 0.005011 | 0.005499 | **0.006899** |
| MRR@10 | 0.000017 | 0.004321 | 0.004696 | **0.005565** |
| 唯一推荐 Semantic ID | 87,862 | 10 | 1,270 | **2,248** |
| 目录覆盖率 | 23.418% | 0.0027% | 0.338% | **0.599%** |

相对 `EXP-006`，两轮训练多命中 32 位测试用户，HR@10 相对提升 39.51%，NDCG@10 相对提升 25.45%，MRR@10 相对提升 18.50%；唯一推荐 SID 增加 978 个，目录覆盖率相对提升 77.01%。相对 MostPopular，HR@10 高 0.40 个百分点，多命中 40 位用户，差距已由 `EXP-006` 的 0.08 个百分点扩大。

这些变化说明增加训练轮数同时改善了相关性和候选多样性，模型对个性化信号的利用比单轮训练更明显。不过测试 HR@10 的绝对值仍只有 1.13%，覆盖率仍不足 0.6%，不能据此认为候选集中问题已经解决。

## 6. 结论与后续

Qwen2.5-0.5B 两轮训练完整结束，训练过程数值稳定，validation loss 到训练末期仍未反弹；validation 和 test 评估产物均通过完整性检查。与单轮基线相比，主要推荐指标和目录覆盖率均有实质改善，说明当前 0.5B 模型在 1 epoch 时训练不足，将训练延长到 2 epochs 是有效方向。

本实验尚不能确定继续增加到 3 epochs 是否仍有足够收益，也不能把改善完全归因于“多看一轮数据”，因为 cosine 调度周期和 warmup 同时发生变化。下一步应结合已经规划的 3-epoch 或不同学习率实验，比较边际收益与训练成本；同时继续分析推荐频次与商品流行度的相关性。所有排名指标仍为 Semantic-ID 级，测试目标中有 4.33% 存在 SID 碰撞，严格的 item 级效果仍需单独的消歧或重排评估。
