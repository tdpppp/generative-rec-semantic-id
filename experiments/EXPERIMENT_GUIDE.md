# 实验记录与评估约定

本文档维护所有实验共用的项目流程、ID 规则、指标、质量门槛和记录模板。实验全局入口见 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

## 1. 项目流程

```text
TencentGR-1M Parquet
  -> profile 缓存
  -> SASRec：学习商品的 32 维表示
  -> RQ-VAE：量化为多层 Semantic ID
  -> Qwen2 GR：根据序列生成 Semantic ID
```

| 阶段 | 输入 | 输出 | 下一阶段前必须满足 |
|---|---|---|---|
| profile | TencentGR-1M Parquet | 用户序列、特征、ID 映射、缓存 | 映射可追溯 |
| SASRec | 用户序列、item ID、profile 特征 | item embeddings、checkpoint、日志 | 数量一致、ID 唯一、无 NaN/Inf、维度匹配 |
| RQ-VAE | item embeddings | Semantic ID mapping、checkpoint、量化指标 | 覆盖率和 collision rate 达标，codebook 使用正常 |
| Qwen2 GR | 用户序列、Semantic ID mapping | 生成模型、预测结果 | 只能使用通过 RQ-VAE 检查的 mapping |

## 2. 数据与 ID

| 名称 | 用途 | 是否跨阶段保留 |
|---|---|---|
| `local_ids` | profile 内部紧凑 ID | 否 |
| `ids` | 正式 re-indexed item ID，`seq.item_id` 使用它 | 是 |
| `original_ids` | 原始商品 ID，用于对齐和追溯 | 用于追溯 |

RQ-VAE 和 GR 必须使用 `ids`，不能把 `local_ids` 当作最终商品 ID。实验文档应记录数据集、profile、筛选规则、特征组合、缺失处理、ID 映射和映射文件位置；与统一约定相同时可写“沿用默认约定”。

## 3. 指标与质量门槛

SASRec 记录 train/validation loss、`HR@10`、`NDCG@10`、`Recall@10`、embedding shape、NaN/Inf、item ID 覆盖率和唯一率。

RQ-VAE 记录 reconstruction loss、Semantic ID 覆盖率、collision rate、各层 codebook 使用数量和使用率。

GR 记录 train/validation loss、grad norm 有限性、Semantic ID 生成有效率、生成结果的 item 映射有效率，以及 HR、NDCG、Recall 等下游推荐指标。Smoke 至少要求完成预定 steps、成功保存 checkpoint、loss 有限且 grad norm 不出现 `NaN/Inf`。

| 指标 | Smoke | 进入 GR 的当前要求 |
|---|---:|---:|
| embedding NaN/Inf | 必须为 0 | 必须为 0 |
| item ID 覆盖率 | 100% | 100% |
| item ID 唯一率 | 100% | 100% |
| Semantic ID collision rate | 只观察 | `<10%`，最好 `<5%` |
| codebook usage | 只观察 | 不应只有个位数或十位数 code |
| HR/NDCG/Recall | 可暂缺 | 正式推荐结论前必须提供 |

Loss 只能和相同数据、切分、负采样和模型配置的实验比较。RQ-VAE 的 `Epoch/loss_train` 和 `Epoch/loss_recon` 当前是 epoch 内 batch loss 总和，不是均值，只适合在 batch 数相同的 epoch 间比较。

## 4. 实验编号与状态

实验使用递增编号 `EXP-001`、`EXP-002`。文件名使用 `EXP-XXX-short-name.md`。已完成实验保留原始记录；配置发生实质变化时创建新实验，不覆盖旧实验。

| 状态 | 含义 |
|---|---|
| `planned` | 已计划但尚未开始 |
| `running` | 正在运行 |
| `completed` | 已完成，结果可用 |
| `failed` | 运行失败或未达到目标 |
| `inconclusive` | 证据不足以支持明确结论 |
| `abandoned` | 主动停止 |
| `superseded` | 被后续实验替代但保留历史价值 |

## 5. 日志与汇总

```bash
head outputs/10k/sasrec/logs/train.log
tail outputs/10k/sasrec/logs/train.log

source .venv/bin/activate
tensorboard --logdir outputs/10k --host 0.0.0.0 --port 6006

python tools/summarize_experiment.py --run_dir outputs/10k
python tools/summarize_experiment.py \
  --run_dir outputs/10k \
  --output outputs/10k/experiment_summary.json
```

关注 `Loss/train`、`Loss/valid`、`Step/loss_recon`、`Step/loss_total` 和 `Epoch/collision_rate`。单步 SASRec loss 是正样本 BCE 与负样本 BCE 之和，不是准确率。

## 6. 实验文档模板

```markdown
# EXP-XXX：实验名称

[返回实验索引](EXPERIMENT_RECORD.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | EXP-XXX |
| 阶段 / 状态 | ... |
| 前置实验 | ... |
| 数据规模 | ... |
| 是否通过质量门槛 | ... |
| 是否允许进入下一阶段 | ... |
| 核心结论 | ... |

## 2. 目标与变化

目标与假设：
相对于前置实验的变化：
本实验不验证：

## 3. 数据与配置

数据、输入产物、特征、ID 映射：
模型、训练和评估配置：
随机种子：
完整运行命令：
输出目录：

## 4. 执行与产物

运行时长：
峰值显存：
磁盘占用：
是否出现训练崩溃：

产物表：

## 5. 结果

结果和质量检查表：

## 6. 结论与后续

观察：
结论：
结论限制：
后续实验或配置变化：
```

历史数据没有记录的字段写“未记录”，计划实验尚未确定的字段写“待填写”。不根据现有结果反推或估算缺失数据。
