# EXP-009：Qwen2.5-0.5B 10k 两轮训练，学习率 5e-5

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-009` |
| 阶段 / 状态 | Qwen2 GR / `completed` |
| 对照实验 | `EXP-008` |
| 基础模型 | Qwen2.5-0.5B Base |
| 数据规模 | 10,000 users；875,365 train / 10,000 validation / 10,000 test samples |
| 主要变化 | 2 epochs，learning rate 从 `3e-5` 改为 `5e-5`；其余配置与 `EXP-008` 相同 |
| 是否通过训练与产物检查 | 是；训练无非有限 loss/grad norm，validation/test 产物检查均通过 |
| 核心结论 | 更高学习率训练稳定，但测试 HR@10 为 0.98%，低于 `EXP-008` 的 1.13%；`3e-5` 在当前 2-epoch 设置下更好 |

## 2. 目标与实验过程

本实验用于比较 2 epochs 下 `3e-5` 与 `5e-5` 学习率的影响。训练从 Qwen2.5-0.5B 预训练权重重新开始，并非从其他实验 checkpoint 续训，共完成 54,712 steps。

除 learning rate 和输出目录外，数据、模型、BF16、batch size、cosine scheduler、warmup、随机种子和评估口径均与 `EXP-008` 相同。因此本实验主要用于学习率对照，但由于两次都是独立初始化和完整训练，结果应解释为独立重复训练的超参数比较，而不是在同一训练轨迹上修改学习率。

训练结束后加载 validation loss 最低的 step 54,000 权重进行 validation/test 受约束 Top-10 评估。根目录模型与 `checkpoint-54000/model.safetensors` 的 SHA-256 一致，评估使用的是最佳 checkpoint，而非最后的 step 54,712 权重。

## 3. 数据与配置

| 项目 | 内容 |
|---|---|
| profile | `data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82` |
| 训练 / 验证 / 测试样本 | 875,365 / 10,000 / 10,000 |
| Semantic ID mapping | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` |
| mapping 商品数 / 唯一 SID 数 | 386,364 / 375,197 |
| Semantic ID 深度 / 空间 | 3 / `[2048, 2048, 1024]` |
| max sequence length / Top-K | 100 / 10 |
| 评估层级 | Semantic-ID 级；受前缀树约束生成 |
| 模型 / precision | Qwen2.5-0.5B Base / BF16 |
| epochs / total steps | 2 / 54,712 |
| train / eval batch size | 32 / 32 |
| learning rate | `5e-5` |
| scheduler / warmup | cosine / 3% |
| weight decay / max grad norm | 0.01 / 1.0 |
| logging / eval / save steps | 20 / 2,000 / 2,000 |
| best model metric | validation loss，越低越好 |
| 随机种子 | 2025 |
| 配置 | `gr/gr_train_qwen25_05b_10k_ep2_lr5e5_b32.json` |
| 输出目录 | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr5e5_b32/` |

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 开始 / 完成 | 2026-09-24 07:00:46 / 14:48:14（UTC+8） |
| Trainer runtime | 27,657.30 秒，约 7 小时 41 分 |
| 含训练后 validation 评估总时长 | 约 7 小时 47 分 |
| test 评估时长 | 未记录 |
| 峰值显存 | 未记录 |
| checkpoint 与最终模型磁盘占用 | 约 14 GB |
| TensorBoard 磁盘占用 | 约 586 KB |
| 是否出现训练崩溃 | 否；进程 exit code 为 0 |

| 产物 | 路径 | 状态 |
|---|---|---|
| 最佳模型 | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr5e5_b32/model.safetensors` | step 54,000 最佳权重 |
| 最佳 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr5e5_b32/checkpoint-54000/` | validation loss 5.2016 |
| 最终 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr5e5_b32/checkpoint-54712/` | 完整训练状态 |
| 训练日志 / TensorBoard | `train.log` / `outputs/gr/tensorboard/qwen25_05b_10k_ep2_lr5e5_b32/` | 可用 |
| validation 指标 / 预测 | `validation_metrics.json` / `validation_predictions.jsonl` | 10,000 用户，产物检查通过 |
| test 指标 / 预测 | `test_metrics.json` / `test_predictions.jsonl` | 10,000 用户，产物检查通过 |
| 运行状态 | `experiment_status.json` | `completed`，exit code 0 |

## 5. 结果

### 5.1 训练稳定性与验证曲线

训练日志每 20 steps 记录一次，共 2,735 条：

| 指标 | 结果 |
|---|---:|
| 首条 / 末条 train loss | 16.4060 / 5.1577 |
| 最低 / 平均记录 loss | 4.9471 / 5.6118 |
| 首条 / 末条 grad norm | 157.16 / 17.22 |
| 最低 / 最高 grad norm | 10.85 / 305.66 |
| 非有限 loss | 0 / 2,735 |
| 非有限 grad norm | 0 / 2,735 |
| Trainer 汇总 train loss | 5.6118 |

共执行 27 次 validation loss 评估，loss 从 step 2,000 的 6.4075 持续下降至 step 54,000 的 5.2016，没有出现回升。最佳 validation loss 为 5.2016，高于 `EXP-008` 的 4.9557。

### 5.2 Validation 与测试结果

| 指标 | Validation | Test |
|---|---:|---:|
| 用户数 | 10,000 | 10,000 |
| 命中用户 | 109 | 98 |
| HR@10 / Recall@10 | 0.0109（1.09%） | 0.0098（0.98%） |
| NDCG@10 | 0.006598 | 0.005981 |
| MRR@10 | 0.005307 | 0.004826 |
| 合法格式率 | 100% | 100% |
| 合法 mapping 率 | 100% | 100% |
| 唯一推荐 Semantic ID | 2,127 | 2,112 |
| Semantic ID 目录覆盖率 | 0.005669（0.567%） | 0.005629（0.563%） |
| 歧义目标比例 | 4.23% | 4.33% |

validation 和 test 预测均为 10,000 行、10,000 个唯一用户，每位用户 10 个候选；空候选、重复候选和非法候选均为 0，重新计算的命中数与指标文件一致。

### 5.3 与 `EXP-008` 对比

| 指标 | `EXP-008`，2 epochs，3e-5 | `EXP-009`，2 epochs，5e-5 | 变化 |
|---|---:|---:|---:|
| 最佳 validation loss | 4.9557 | 5.2016 | 增加 0.2459 |
| Test 命中用户 | 113 | 98 | 减少 15 |
| Test HR@10 | 1.13% | 0.98% | 相对下降 13.27% |
| Test NDCG@10 | 0.006899 | 0.005981 | 相对下降 13.31% |
| Test MRR@10 | 0.005565 | 0.004826 | 相对下降 13.27% |
| 唯一推荐 SID | 2,248 | 2,112 | 减少 136 |
| Test 目录覆盖率 | 0.599% | 0.563% | 相对下降 6.05% |

更高学习率没有造成数值不稳定，训练仍然完成且所有 loss/grad norm 有限；但 validation loss、测试相关性和候选覆盖率均低于 `EXP-008`。在当前 2-epoch、cosine 调度设置下，`3e-5` 是目前更好的学习率选择。

## 6. 结论与后续

`EXP-009` 证明 `5e-5` 可以稳定训练，但没有优于 `EXP-008` 的 `3e-5`。因此后续相同规模和 2-epoch 配置优先保留 `3e-5`，除非新的实验同时改变训练轮数、调度或其他关键变量。

这次对照仍是 Semantic-ID 级评估；测试目标歧义率为 4.33%，严格 item 级指标仍需碰撞消歧或重排。下一步可继续比较 3 epochs 配置，但应同时关注 validation loss 后期平台、推荐指标边际收益和额外约 7-8 小时训练成本。
