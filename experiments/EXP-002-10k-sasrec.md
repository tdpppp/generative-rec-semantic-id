# EXP-002：10k SASRec

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-002` |
| 阶段 / 状态 | SASRec / `completed` |
| 前置实验 | `EXP-001` |
| 数据规模 | 10,000 users / 386,364 items |
| 是否通过质量门槛 | 通过 embedding 工程检查；推荐效果未正式验证 |
| 是否允许进入下一阶段 | embedding 可进入 RQ-VAE |
| 核心结论 | loss 下降并成功导出有效 embedding，但缺少 HR/NDCG/Recall |

## 2. 目标与变化

目标是观察 10k 用户规模下 SASRec 的真实训练行为，并导出供 RQ-VAE 使用的 item embedding。相对于 `EXP-001`，数据规模从 smoke 扩展到 10k 用户。本实验不验证最终推荐效果。

## 3. 数据与配置

| 项目 | 内容 |
|---|---|
| 数据集 | TencentGR-1M |
| 用户数 / 商品数 | 10,000 / 386,364 |
| profile 缓存 | 858 MB |
| ID 约定 | `ids` 跨阶段传递，`local_ids` 仅用于 profile 内部计算 |
| train steps | 32 |
| 随机种子 | 未记录 |
| 完整训练配置和命令 | 当前历史记录未保留 |
| 输出目录 | `outputs/10k/` |

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 运行时长 | 未记录 |
| 峰值显存 | 未记录 |
| 磁盘占用 | 未记录 |
| 是否出现训练崩溃 | 否 |

| 产物 | 路径 | 用途 | 状态 |
|---|---|---|---|
| train log | `outputs/10k/sasrec/logs/train.log` | 训练过程 | 可用 |
| TensorBoard | `outputs/10k/sasrec/tensorboard/` | 曲线分析 | 可用 |
| checkpoint | `outputs/10k/sasrec/checkpoints/global_step32.valid_loss=1.2027/` | 模型恢复 | 可用 |
| embeddings | `outputs/10k/emb/embeddings.npz` | RQ-VAE 输入 | 可用 |

## 5. 结果

| 指标 | 结果 |
|---|---:|
| 81 有效 / 缺失 | 363,712 / 22,652 |
| 82 有效 / 缺失 | 381,468 / 4,896 |
| train loss | 2.0001 -> 1.2299 |
| 平均 train loss | 1.4404 |
| validation loss | 1.2027 |
| 导出向量 | 386,364 x 32, float32 |
| ID 唯一性 | 全部唯一 |
| NaN/Inf | 全部有限 |
| HR@10 / NDCG@10 / Recall@10 | 尚未产出 |

## 6. 结论与后续

SASRec loss 明显下降，Stage 1 工程链路正常。由于尚未产出 `HR@10`、`NDCG@10` 和 `Recall@10`，不能声称推荐效果已经得到正式验证。embedding 通过 shape、ID 唯一性和有限性检查，可作为 `EXP-003` 的输入。
