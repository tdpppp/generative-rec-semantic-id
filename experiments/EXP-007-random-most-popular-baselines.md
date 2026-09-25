# EXP-007：Random / MostPopular 推荐基线

[返回实验索引](EXPERIMENT_RECORD.md) | [查看统一约定](EXPERIMENT_GUIDE.md)

## 1. 实验摘要

| 项目 | 内容 |
|---|---|
| 实验 ID | `EXP-007` |
| 阶段 / 状态 | GR 评估基线 / `completed` |
| 前置实验 | `EXP-004`、`EXP-006` |
| 数据规模 | 10,000 test users；每位用户推荐 10 个 Semantic ID |
| 对照方法 | Random、MostPopular |
| 是否通过评估质量检查 | 是；预测数量完整、候选无重复、格式与 mapping 有效率均为 100% |
| 核心结论 | Qwen2.5-0.5B GR 的 HR@10 高于 MostPopular，但优势只有 0.08 个百分点；Random 证明高覆盖率本身不代表有效推荐，MostPopular 则揭示当前数据存在较强的热门度信号 |

## 2. 目标与评估方法

本实验为 `EXP-006` 的 Qwen2.5-0.5B GR 结果补充两个无需训练的推荐基线。Random 用于给出从完整 Semantic ID 目录均匀采样时的随机下界；MostPopular 用于衡量只依赖训练历史中全局热门度、完全忽略用户兴趣时可以达到的效果。

两个基线共享同一套测试数据、Semantic ID mapping、Top-10 评估代码和指标口径，因此合并记录为一个实验。MostPopular 的流行度只统计每条用户序列中验证和测试留出项之前的事件，即 `sequence[:-2]`，不会使用验证目标或测试目标；候选按出现次数降序排列，次数相同时按 Semantic ID 字典序排列。Random 从完整且去重后的 Semantic ID 目录中为每位用户无放回采样 10 个候选，固定随机种子为 2025。

本实验和 `EXP-006` 一样在 Semantic-ID 层级计算排名指标。它不处理一个 Semantic ID 对应多个 item 时的消歧问题，也不提供严格的 item 级推荐结论。

## 3. 数据与配置

| 项目 | 内容 |
|---|---|
| profile | `data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82` |
| split / 用户数 | test / 10,000 |
| Semantic ID mapping | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` |
| 唯一 Semantic ID 数 | 375,197 |
| Semantic ID 深度 | 3 |
| Top-K | 10 |
| max sequence length | 100 |
| 评估层级 | Semantic-ID 级 |
| Random 随机种子 | 2025 |
| MostPopular 统计范围 | 所有用户的训练历史，不含 validation 和 test holdout |
| 评估程序 | `gr/evaluate_gr_baselines.py` |
| 输出目录 | `outputs/gr/baselines/10k_balanced/semantic_id_top10/` |

## 4. 执行与产物

| 项目 | 结果 |
|---|---|
| 完成日期 | 2026-09-23 |
| 运行时长 | 未记录 |
| 峰值显存 | 未记录；两个方法无需模型训练或神经网络推理 |
| 输出磁盘占用 | 约 7.6 MB |
| 是否出现训练崩溃 | 不适用；两个基线均无需训练，评估正常完成 |

| 产物 | 路径 | 状态 |
|---|---|---|
| 汇总对比 | `outputs/gr/baselines/10k_balanced/semantic_id_top10/comparison.json` | 可用 |
| MostPopular 指标 | `outputs/gr/baselines/10k_balanced/semantic_id_top10/most_popular_metrics.json` | 可用 |
| MostPopular 预测 | `outputs/gr/baselines/10k_balanced/semantic_id_top10/most_popular_predictions.jsonl` | 10,000 行 |
| Random 指标 | `outputs/gr/baselines/10k_balanced/semantic_id_top10/random_metrics.json` | 可用 |
| Random 预测 | `outputs/gr/baselines/10k_balanced/semantic_id_top10/random_predictions.jsonl` | 10,000 行 |

## 5. 结果

### 5.1 基线结果

| 指标 | Random | MostPopular |
|---|---:|---:|
| 测试用户 | 10,000 | 10,000 |
| 命中用户 | 1 | 73 |
| HR@10 | 0.0001（0.01%） | 0.0073（0.73%） |
| Recall@10 | 0.0001（0.01%） | 0.0073（0.73%） |
| NDCG@10 | 0.000036 | 0.005011 |
| MRR@10 | 0.000017 | 0.004321 |
| 合法格式率 | 100% | 100% |
| 合法 mapping 率 | 100% | 100% |
| 唯一推荐 Semantic ID | 87,862 | 10 |
| Semantic ID 目录覆盖率 | 0.234176（23.418%） | 0.000027（0.0027%） |
| 歧义目标比例 | 4.33% | 4.33% |

Random 在 100,000 个推荐位置中覆盖 87,862 个 Semantic ID，但只命中 1 位用户。这说明目录覆盖率必须与相关性指标一起解释：高覆盖可以来自接近均匀的随机分散，并不表示推荐有效。

MostPopular 对所有用户返回相同的 10 个热门 Semantic ID，目录覆盖率因此只有约 0.0027%，但命中 73 位用户，HR@10 达到 0.73%。它远高于 Random，说明测试目标中存在明显的全局热门度信号；同时，它完全没有个性化能力，不能作为目标方案。

### 5.2 与 Qwen2.5-0.5B GR 对比

| 指标 | Random | MostPopular | Qwen2.5-0.5B GR (`EXP-006`) |
|---|---:|---:|---:|
| 命中用户 | 1 | 73 | 81 |
| HR@10 | 0.01% | 0.73% | 0.81% |
| NDCG@10 | 0.000036 | 0.005011 | 0.005499 |
| MRR@10 | 0.000017 | 0.004321 | 0.004696 |
| 唯一推荐 Semantic ID | 87,862 | 10 | 1,270 |
| Semantic ID 目录覆盖率 | 23.418% | 0.0027% | 0.338% |

GR 的 HR@10 比 MostPopular 高 0.08 个百分点，即多命中 8 位用户；NDCG@10 和 MRR@10 也分别高约 9.7% 和 8.7%。这表明模型已经略微超过纯热门度策略，但优势较小，尚不足以证明它充分利用了用户历史进行个性化推荐。

GR 覆盖 1,270 个 Semantic ID，明显高于 MostPopular 的 10 个，但仍只占目录的 0.338%，而且远低于 Random 的分散程度。结合效果与覆盖率来看，当前模型不是纯粹复现固定 Top-10 热门列表，但候选分布仍然高度集中。

## 6. 结论与后续

Random 和 MostPopular 已建立可重复的 Semantic-ID 级 Top-10 参照。Qwen2.5-0.5B GR 明显优于 Random，却只小幅优于 MostPopular，因此 `EXP-006` 中“推荐相关性与多样性不足”的判断得到进一步支持。后续实验应至少同时报告这两个基线，重点分析 GR 推荐频次与训练集流行度的相关性，确认候选集中究竟主要来自热门度偏置、训练不足还是解码策略。

当前结果仍受 Semantic ID 碰撞影响：三种方法面对相同的 4.33% 歧义目标，所有 HR、NDCG、Recall 和 MRR 均是 Semantic-ID 级指标。完成 item 级消歧或重排之前，不能把表中结果直接解释为严格的 item 级推荐性能。
