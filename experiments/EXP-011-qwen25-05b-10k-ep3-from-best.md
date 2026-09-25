# EXP-011：Qwen2.5-0.5B 从两轮最佳模型续训 1 epoch

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-011` |
| 阶段 / 状态 | Qwen2 GR / `completed` |
| 前置实验 | `EXP-008` |
| 基础模型 | `EXP-008` 的最佳 `checkpoint-54000` |
| 数据规模 | 10,000 users；875,365 train / 10,000 validation samples |
| 主要变化 | 从两轮最佳权重继续训练 1 epoch；使用新的 optimizer 和 cosine scheduler，learning rate 为 `1e-5` |
| 是否通过训练与产物检查 | 是；训练完成，最佳模型与两个 checkpoint 完整，loss 和 grad norm 均为有限值 |
| 是否通过 validation 检查 | 是；两次完整评估结果逐行一致，10,000 用户均有 10 个合法且无重复的候选 |
| 核心结论 | validation HR@10 从 1.25% 提升至 1.32%，唯一推荐 SID 从 2,249 增至 3,127；提升存在但幅度有限，尚未运行 test |

## 2. 目标与实验过程

本实验检验在 `EXP-008` 的两轮最佳权重基础上，以较低学习率追加训练是否还能改善 Semantic-ID 级推荐效果。初始化权重来自：

`outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/checkpoint-54000`

训练没有恢复旧 checkpoint 中的 optimizer、scheduler 或 global step，而是使用新 optimizer 和新 cosine scheduler 完成 1 epoch。因此它可以理解为“从两轮最佳模型进行一次低学习率续训”，但不是严格意义上保持同一优化器和调度器连续训练得到的 3-epoch 单变量对照。

训练完成 27,356 steps。validation loss 最低点出现在 step 26,000，训练结束后 `load_best_model_at_end` 加载并保存了该权重。根目录 `model.safetensors` 与 `checkpoint-26000/model.safetensors` 的 SHA-256 完全一致。

训练后使用 `max_seq_length=100`、`top_k=10`、`num_beams=10` 在 validation split 上完成受约束生成评估。流水线先后进行了两次完整评估；两份 metrics 和 predictions 完全相同。流水线中的 artifact checker 因运行环境未正确解析 `gr` 模块而报 `No module named 'gr'`，但评估本身已经成功完成，本文又独立检查了预测行数、用户数、候选数量、重复候选和命中数。

## 3. 数据与配置

### 3.1 数据与评估口径

| 项目 | 内容 |
|---|---|
| profile | `data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82` |
| 用户数 | 10,000 |
| 训练 / validation 样本 | 875,365 / 10,000 |
| Semantic ID mapping | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` |
| mapping 商品数 / 唯一 SID 数 | 386,364 / 375,197 |
| Semantic ID 深度 / 空间 | 3 / `[2048, 2048, 1024]` |
| 训练 / 评估 max sequence length | 100 / 100 |
| 评估层级 / Top-K | Semantic-ID 级 / 10 |
| beam width | 10 |
| 解码约束 | Semantic ID 前缀树；只允许生成 mapping 中存在的三层组合 |

### 3.2 模型与训练配置

| 项目 | 内容 |
|---|---|
| 初始化 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/checkpoint-54000` |
| tokenizer | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32` |
| optimizer / scheduler 状态 | 不恢复，重新初始化 |
| precision | BF16 |
| epochs / total steps | 1 / 27,356 |
| train / eval batch size | 32 / 32 |
| gradient accumulation | 1 |
| learning rate | 1e-5 |
| scheduler / warmup | cosine / 3% |
| weight decay / max grad norm | 0.01 / 1.0 |
| logging / eval / save steps | 20 / 2,000 / 2,000 |
| best model metric | validation loss，越低越好 |
| save limit | 2 checkpoints |
| dataloader workers | 4 |
| 随机种子 | 2025 |
| 配置 | `gr/gr_train_qwen25_05b_10k_ep3_from_best_lr1e5_b32.json` |
| 输出目录 | `outputs/gr/checkpoints/qwen25_05b_10k_ep3_from_best_lr1e5_b32/` |

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 完成日期 | 2026-09-25（UTC+8） |
| Trainer runtime | 13,594.69 秒，约 3 小时 46 分 35 秒 |
| 单次 validation 推理时长 | 约 6 分 10 秒 |
| validation 最大 CUDA allocated memory | 2,189,426,176 bytes，约 2.04 GiB |
| 训练峰值显存 | 未记录 |
| checkpoint 与最终模型磁盘占用 | 约 14 GB |
| TensorBoard 磁盘占用 | 约 295 KB |
| 是否出现训练崩溃 | 否；完整训练至 step 27,356 |

| 产物 | 路径 | 状态 |
|---|---|---|
| 最佳模型 | `outputs/gr/checkpoints/qwen25_05b_10k_ep3_from_best_lr1e5_b32/model.safetensors` | step 26,000 最佳权重 |
| 最佳 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_ep3_from_best_lr1e5_b32/checkpoint-26000/` | validation loss 4.7459 |
| 最终 checkpoint | `outputs/gr/checkpoints/qwen25_05b_10k_ep3_from_best_lr1e5_b32/checkpoint-27356/` | 完整训练状态 |
| 训练日志 | `outputs/gr/checkpoints/qwen25_05b_10k_ep3_from_best_lr1e5_b32/train.log` | 可用 |
| TensorBoard | `outputs/gr/tensorboard/qwen25_05b_10k_ep3_from_best_lr1e5_b32/` | 可用 |
| validation metrics | `outputs/gr/ep3_history50_pipeline/attempts/ep3_validation_maxlen100/attempt-1-20260925T004528/metrics.json` | 10,000 用户，完整 |
| validation predictions | `outputs/gr/ep3_history50_pipeline/attempts/ep3_validation_maxlen100/attempt-1-20260925T004528/predictions.jsonl` | 10,000 行，完整性检查通过 |
| 重复 validation 对照 | `outputs/gr/ep3_history50_pipeline/attempts/ep3_validation_maxlen100/attempt-2-20260925T005235/` | 与第一次逐字节一致 |

误启动的第三次重复 validation 已终止。它在 `outputs/gr/ep3_manual_validation_20260925/` 留下未完成的临时 predictions，不作为正式实验产物，也不参与本文指标统计。

## 5. 结果

### 5.1 训练稳定性与验证曲线

训练日志每 20 steps 记录一次，共 1,367 条：

| 指标 | 结果 |
|---|---:|
| 首条 / 末条 train loss | 4.6644 / 4.4587 |
| 最低 / 最高 / 平均记录 loss | 4.3264 / 4.9271 / 4.6466 |
| 首条 / 末条 grad norm | 23.81 / 28.34 |
| 最低 / 最高 grad norm | 22.68 / 32.25 |
| 非有限 loss | 0 / 1,367 |
| 非有限 grad norm | 0 / 1,367 |
| Trainer 汇总 train loss | 4.6464 |

共执行 13 次 validation loss 评估，关键节点如下：

| Step | Epoch | Validation loss |
|---:|---:|---:|
| 2,000 | 0.07 | 4.9976 |
| 6,000 | 0.22 | 4.9511 |
| 10,000 | 0.37 | 4.8889 |
| 14,000 | 0.51 | 4.8341 |
| 18,000 | 0.66 | 4.7835 |
| 22,000 | 0.80 | 4.7555 |
| 24,000 | 0.88 | 4.7480 |
| 26,000 | 0.95 | **4.7459** |

最佳 validation loss 相比初始化来源 `EXP-008` 的 4.9557 下降 0.2098，降幅约 4.23%。训练后段仍在改善，但 step 24,000 到 26,000 仅下降 0.0021，已经接近平台区间。

### 5.2 Semantic-ID validation 结果

| 指标 | EXP-008 | EXP-011 | 变化 |
|---|---:|---:|---:|
| 命中用户 | 125 | **132** | +7 |
| HR@10 / Recall@10 | 0.0125（1.25%） | **0.0132（1.32%）** | +0.07 个百分点 |
| NDCG@10 | 0.007394 | **0.007582** | +2.55% |
| MRR@10 | 0.005841 | **0.005866** | +0.43% |
| 合法格式率 | 100% | 100% | 持平 |
| 合法 mapping 率 | 100% | 100% | 持平 |
| 唯一推荐 Semantic ID | 2,249 | **3,127** | +878 |
| Semantic ID 目录覆盖率 | 0.5994% | **0.8334%** | +0.2340 个百分点 |
| 歧义目标比例 | 4.23% | 4.23% | 持平 |

两次完整评估均包含 10,000 行、10,000 个唯一用户，每位用户恰有 10 个候选；空候选、候选数异常和列表内重复均为 0。重新统计得到 132 次命中，与 metrics 一致；两份 predictions 的 SHA-256 均为 `3a0ea8711b3eb396994527b64289382c697afb505e4dfa207756f78bc3ff1446`。

与 `EXP-008` 相比，本实验的 validation HR@10 相对提升 5.60%，NDCG@10 相对提升 2.55%，MRR@10 基本持平。更明显的变化是唯一推荐 SID 增加 878 个，覆盖率相对提升约 39.04%。这说明低学习率续训继续扩大了候选分布，但相关性指标的边际收益已经明显小于从 1 epoch 增加到 2 epochs 时的收益。

## 6. 结论与后续

本实验训练和 Semantic-ID validation 均已完成。低学习率续训使 validation loss、HR@10、NDCG@10 和候选覆盖率继续改善，其中覆盖率提升最明显；但 MRR@10 仅小幅变化，相关性收益已经接近边际递减。

当前不应仅根据 validation 结果立即对多个配置运行 test。后续应先完成计划中的 history50 对照和 beam search validation，再按照 validation NDCG@10、HR@10、MRR@10 的顺序选择一个最终配置，只对最终配置运行 test。

所有排名指标仍为 Semantic-ID 级，4.23% 的 validation 目标存在 SID 碰撞。严格的 item 级推荐效果仍需要单独的消歧或重排评估。
