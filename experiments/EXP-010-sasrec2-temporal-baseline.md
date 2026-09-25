# EXP-010：SASRec2 时间切分 10k 推荐基线

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-010` |
| 阶段 / 状态 | SASRec 推荐基线 / `completed` |
| 前置实验 | `EXP-002`、`EXP-004`、`EXP-006` |
| 数据规模 | 10,000 users；386,364 items；train 875,365 samples；validation/test 各 10,000 users |
| 主要目的 | 使用与 Qwen 相同的时间留出协议，建立 item 级 SASRec Top-10 基线 |
| 模型 | SASRec2，多模态特征 81/82，hidden size 32 |
| 是否通过评估质量检查 | 是；validation/test 均完整评估 10,000 用户，Top-10 无重复，预测产物可复核 |
| 核心结论 | Test item-level HR@10 为 0.52%，NDCG@10 为 0.002697；该结果提供了时间切分下的 SASRec item 级参照，但不能直接与 Qwen 的 Semantic-ID 级指标比较 |

## 2. 目标与实验过程

`EXP-002` 主要验证了旧 SASRec 链路和 embedding 导出，缺少正式推荐指标。本实验使用独立的 `sasrec2` 实现，保持原 `sasrec/` 入口不变，并采用与 Qwen 实验一致的 temporal holdout：训练样本使用历史前缀预测后续行为，validation 预测倒数第二个商品，test 预测最后一个商品。

训练使用多模态商品特征 81/82，完成 3 个 epoch。每个 epoch 保存一个 checkpoint，并以 validation loss 最低的模型作为 `best_model.pt`。随后使用该最佳模型对 validation 和 test 的完整 10,000 用户集合进行全商品 item-level Top-10 排序评估。

本实验输出的是具体 item 的排名，不经过 Semantic ID mapping；因此它可以作为 item-level SASRec 参照，但不能把它和 `EXP-006` 至 `EXP-009` 的 Semantic-ID-level HR/NDCG/MRR 直接排成同一张性能排行榜。

## 3. 数据与配置

| 项目 | 内容 |
|---|---|
| 原始数据 | `data/TencentGR_1M` |
| profile 缓存 | `data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82` |
| 用户数 / 商品数 | 10,000 / 386,364 |
| train / validation / test | 875,365 / 10,000 / 10,000 samples |
| 时间切分 | train 使用每个序列的历史前缀；validation 目标为倒数第二个商品；test 目标为最后一个商品 |
| 多模态特征 | `81`、`82` |
| max sequence length | 100 |
| hidden units / attention heads / blocks | 32 / 1 / 1 |
| dropout | 0.2 |
| batch size | 256 |
| epochs | 3 |
| learning rate | 1e-3 |
| optimizer | Adam，betas=(0.9, 0.98) |
| loss | 正样本与负样本 BCE loss 之和 |
| max grad norm | 1.0 |
| 随机种子 | 2025 |
| DataLoader workers | 4 |
| 评估 Top-K | 10 |
| 评估粒度 | item-level |
| 训练入口 | `sasrec2/train.py` |
| 评估入口 | `sasrec2/evaluate.py` |
| 输出目录 | `outputs/sasrec2/temporal_10k_mm81_82/` |

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 运行时间 | 未精确记录；从 epoch checkpoint 时间看，训练约在 2026-09-24 19:00 至 20:28（UTC+8）完成 |
| validation 评估耗时 | 约 70.2 秒 |
| test 评估耗时 | 约 69.1 秒 |
| 峰值显存 | 未记录 |
| 输出目录磁盘占用 | 约 1.4 GB |
| 是否出现训练崩溃 | 未发现；epoch 1-3 checkpoint 和最佳模型均已生成 |
| train loss 日志 | 未单独保存，无法复原完整 train loss 曲线 |

| 产物 | 路径 | 状态 |
|---|---|---|
| Epoch 1 checkpoint | `outputs/sasrec2/temporal_10k_mm81_82/checkpoints/epoch-1.valid_loss=0.003535.pt` | 可用 |
| Epoch 2 checkpoint | `outputs/sasrec2/temporal_10k_mm81_82/checkpoints/epoch-2.valid_loss=0.003259.pt` | 可用 |
| Epoch 3 checkpoint | `outputs/sasrec2/temporal_10k_mm81_82/checkpoints/epoch-3.valid_loss=0.003217.pt` | 可用 |
| 最佳模型 | `outputs/sasrec2/temporal_10k_mm81_82/checkpoints/best_model.pt` | epoch 3，validation loss 0.003217 |
| validation 指标 / 预测 | `validation_metrics.json` / `validation_predictions.jsonl` | 10,000 用户 |
| test 指标 / 预测 | `test_metrics.json` / `test_predictions.jsonl` | 10,000 用户 |
| 训练代码说明 | `sasrec2/README.md` | 记录训练与评估命令 |

## 5. 结果

### 5.1 Validation loss

| Epoch | Validation loss |
|---:|---:|
| 1 | 0.003535 |
| 2 | 0.003259 |
| 3 | **0.003217** |

validation loss 连续下降，epoch 3 被保存为最佳模型。由于训练 stdout 没有保存为独立日志文件，本实验不记录 train loss 的首值、末值或均值。

### 5.2 Item-level Top-10 结果

| 指标 | Validation | Test |
|---|---:|---:|
| 用户数 | 10,000 | 10,000 |
| 命中用户 | 70 | 52 |
| HR@10 | 0.0070（0.70%） | 0.0052（0.52%） |
| Recall@10 | 0.0070（0.70%） | 0.0052（0.52%） |
| NDCG@10 | 0.003861 | 0.002697 |
| MRR@10 | 0.002920 | 0.001943 |
| Top-10 内重复 item | 0 | 0 |
| 预测行数 | 10,000 | 10,000 |

validation 命中 rank 分布为：rank 1 有 17 次、rank 2 有 8 次、rank 3 有 4 次、rank 4 有 6 次、rank 5 有 11 次、rank 6 有 8 次、rank 7 有 2 次、rank 8 有 5 次、rank 9 有 2 次、rank 10 有 7 次。test 命中 rank 分布为：rank 1 有 9 次、rank 2 有 7 次、rank 3 有 5 次、rank 4 有 9 次、rank 5 有 3 次、rank 6 有 5 次、rank 7 有 1 次、rank 8 有 4 次、rank 9 有 3 次、rank 10 有 6 次。

### 5.3 与 GR 结果的解释边界

本实验的 test HR@10 为 0.52%，但这个数值不能直接和 `EXP-006` 的 0.81%、`EXP-008` 的 1.13% 或 `EXP-009` 的 0.98% 比较，因为 SASRec2 直接判断预测 item 是否等于目标 item，而 GR 实验判断的是预测 Semantic ID 是否等于目标 Semantic ID。GR 的 Semantic ID 碰撞会使两种指标的命中事件不同。

本实验更适合回答：在相同 temporal holdout 数据上的 item-level SASRec 能达到什么水平。后续若要进行严格模型比较，需要将 GR 的 Semantic ID 预测展开到 item，并使用同一 item-level 评估协议，或者把 SASRec 的 item 预测转换到 Semantic-ID 层级后再比较。

### 5.4 与前序实验的指标变化与分析

先列出各次正式 Top-10 评估的 test 结果。`EXP-007` 至 `EXP-009` 是 Semantic-ID-level 指标，`EXP-010` 是 item-level 指标，因此表中数值可以用于观察变化，但不能直接作为同一尺度的模型排名。

| 实验 / 方法 | 命中用户 | HR@10 | NDCG@10 | MRR@10 | 唯一推荐对象 / 覆盖率 |
|---|---:|---:|---:|---:|---:|
| `EXP-007` Random | 1 | 0.01% | 0.000036 | 0.000017 | 87,862 SID / 23.418% |
| `EXP-007` MostPopular | 73 | 0.73% | 0.005011 | 0.004321 | 10 SID / 0.0027% |
| `EXP-006` GR，1 epoch | 81 | 0.81% | 0.005499 | 0.004696 | 1,270 SID / 0.338% |
| `EXP-008` GR，2 epochs，3e-5 | 113 | **1.13%** | **0.006899** | **0.005565** | 2,248 SID / 0.599% |
| `EXP-009` GR，2 epochs，5e-5 | 98 | 0.98% | 0.005981 | 0.004826 | 2,112 SID / 0.563% |
| `EXP-010` SASRec2，item-level | 52 | 0.52% | 0.002697 | 0.001943 | item Top-10 / 不适用 |

从相邻 GR 实验的指标变化看，`EXP-008` 相比 `EXP-006` 的 1 epoch 有明显改善：HR@10 从 0.81% 增至 1.13%，增加 0.32 个百分点，相对提升 39.51%；NDCG@10 相对提升 25.45%，MRR@10 相对提升 18.50%，唯一推荐 SID 增加 978 个，目录覆盖率相对提升 77.01%。这说明在当前配置下，1 epoch 可能仍未充分训练，增加到 2 epochs 同时改善了相关性和候选多样性。

`EXP-009` 将学习率从 `3e-5` 提高到 `5e-5` 后，训练仍然稳定，但 test HR@10 从 1.13% 降至 0.98%，减少 0.15 个百分点，相对下降 13.27%；NDCG@10 和 MRR@10 也分别下降 13.31% 和 13.27%，目录覆盖率下降 6.05%。因此当前证据支持 `3e-5` 比 `5e-5` 更适合这组 2-epoch 训练，但还不能推出它是所有 epoch 数和调度方式下的全局最优学习率。

和 `EXP-007` 的两个无训练基线相比，`EXP-006` 的 GR 只比 MostPopular 高 0.08 个百分点；延长到 `EXP-008` 后，差距扩大到 0.40 个百分点，命中用户从 73 增加到 113。这说明延长训练后，GR 利用用户历史的收益更加明显，而不是只复现固定的热门 SID 列表。不过 GR 的指标仍然是 SID 命中，不能据此直接断言它在具体 item 命中上也同样超过 MostPopular。

`EXP-010` 自身的 validation 到 test 变化是：HR@10 从 0.70% 降至 0.52%，减少 0.18 个百分点；NDCG@10 从 0.003861 降至 0.002697；MRR@10 从 0.002920 降至 0.001943。GR 的对应落差为：`EXP-008` 的 HR@10 从 1.25% 降至 1.13%，`EXP-009` 从 1.09% 降至 0.98%。这些 validation-test 差异表明 test 难度更高或存在一定泛化落差，但不能单独归因于过拟合，因为各实验的模型、表示空间和命中定义并不完全相同。

因此，当前可以确定的趋势有三点：训练从 1 epoch 延长到 2 epochs 对 GR 有益；在 2 epochs 下把学习率从 `3e-5` 提高到 `5e-5` 反而有害；SASRec2 在具体 item 空间的 test HR@10 为 0.52%。但最后一点不能直接与 GR 的 1.13% 或 0.98% 比较，严格比较仍需要把 GR 的 SID 预测展开为 item 并处理 SID 碰撞。

## 6. 结论与后续

SASRec2 时间切分基线训练完成，validation loss 从 0.003535 降至 0.003217，最佳模型成功用于 validation/test 全量评估。test item-level HR@10 为 0.52%，NDCG@10 为 0.002697，说明在当前 10k temporal protocol 下，单独的 SASRec item-level 相关性仍然较低。

与前序实验相比，本实验的主要贡献是补齐了一个可复核的 temporal、item-level SASRec 推荐基线，而不是证明 SASRec 已经优于或劣于 GR。当前结果应作为后续统一评估的参照点：如果要回答 GR 是否真正优于 SASRec，必须先对 GR 生成的 SID 做 item 级展开和碰撞消歧，再使用相同的目标 item、候选数量和时间切分计算指标。在此之前，GR 的 Semantic-ID-level 结果和本实验的 item-level 结果只能分别解释，不能直接合并成一个模型排名结论。

该实验没有产生新的 RQ-VAE 输入 embedding，因此不替代 `EXP-002` 的 embedding 训练记录。后续应保留 item-level 与 Semantic-ID-level 两套指标，并优先实现 SID 碰撞后的 item 级重排或消歧评估；完成统一口径后，再决定是否继续扩大 SASRec 模型或将训练资源投入 GR 的 epoch、学习率和解码策略比较。
