# EXP-003：10k RQ-VAE Baseline

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-003` |
| 阶段 / 状态 | RQ-VAE / `failed` |
| 前置实验 | `EXP-002` |
| 数据规模 | 386,364 items |
| 是否通过质量门槛 | 否 |
| 是否允许进入下一阶段 | 否，不得进入 GR |
| 核心结论 | codebook 严重塌缩，Semantic ID mapping 不可用 |

## 2. 目标与变化

目标是首次验证将 SASRec 的 32 维 item embedding 量化为三层 Semantic ID 的可行性。本实验是 RQ-VAE 首次 baseline，不验证 GR 效果。

## 3. 数据与配置

| 项目 | 内容 |
|---|---|
| 输入商品数 | 386,364 |
| 输入产物 | `outputs/10k/emb/embeddings.npz` |
| 随机种子 | 未记录 |
| 输出目录 | `outputs/10k/rqvae/`、`outputs/10k/emb_infer/` |

```text
epochs=20
batch_size=4096
warmup_epochs=50
eval_step=20
num_emb_list=[2048, 2048, 1024]
sk_epsilons=[0.0, 0.0, 0.0]
```

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 运行时长 | 未记录 |
| 峰值显存 | 未记录 |
| 磁盘占用 | 未记录 |
| 是否出现训练崩溃 | 否 |

| 产物 | 路径 | 用途 | 状态 |
|---|---|---|---|
| TensorBoard | `outputs/10k/rqvae/tensorboard/` | loss 和 collision 曲线 | 可查看 |
| checkpoint | `outputs/10k/rqvae/checkpoints/09-22-2026/` | 模型恢复 | 失败对照 |
| Semantic ID mapping | `outputs/10k/emb_infer/sinkhorn/worker_0_output.txt` | item 到 Semantic ID | 禁止用于 GR |

## 5. 结果

| 指标 | 结果 |
|---|---:|
| reconstruction epoch sum | 3.4252 -> 1.0720 |
| 训练期碰撞率 | 99.51% |
| 推理后唯一 Semantic ID | 3,552 |
| 推理后碰撞商品 | 382,812 |
| 推理后碰撞率 | 99.08% |
| 第一层 code 使用 | 12 / 2,048 |
| 第二层 code 使用 | 20 / 2,048 |
| 第三层 code 使用 | 235 / 1,024 |

## 6. 结论与后续

观察到 reconstruction loss 改善，但各层 codebook 使用严重不足，推理后碰撞率达到 99.08%。

可能原因：

- 实际训练 20 epoch，小于 50 epoch warmup；
- collision 只在最后评估一次；
- 训练阶段未启用 Sinkhorn 平衡。

本实验未通过质量门槛。mapping 不可用于 Qwen2 GR，仅保留为失败对照。后续由 `EXP-004` 验证调整 warmup、启用 Sinkhorn 和提高评估频率后的效果。
