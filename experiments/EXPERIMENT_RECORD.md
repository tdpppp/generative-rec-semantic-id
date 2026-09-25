# TencentGR-1M 实验记录

本文档是项目实验的全局入口，用于快速查看当前状态、实验结论和关键产物。统一记录规则见 [实验记录与评估约定](EXPERIMENT_GUIDE.md)，每次实验的配置、结果和分析见对应的独立实验文档。

## 1. 当前状态

| 项目 | 当前状态 |
|---|---|
| 最后更新 | 2026-09-25 |
| 当前阶段 | Qwen2.5-0.5B 从两轮最佳 checkpoint 低学习率续训及 Semantic-ID validation 已完成 |
| 当前关注 | 比较训练/评估历史长度与 beam width，随后只对最终 validation 最优配置运行 test |
| 当前阻塞问题 | SASRec2 为 item-level，GR 为 Semantic-ID-level；两者尚不能直接比较，GR test 也尚未选择最终配置 |
| GR 状态 | 续训后 validation HR@10 为 1.32%，NDCG@10 为 0.7582%，目录覆盖率提升至 0.833%；test 尚未运行 |
| 下一项实验 | 完成 history50 对照和 beam search validation，选择最终配置 |

## 2. 实验索引

| ID | 阶段 | 实验 | 数据规模 | 状态 | 核心结论 | 文档 |
|---|---|---|---:|---|---|---|
| `EXP-001` | Pipeline | Smoke test | 256 users | `completed` | 工程链路可运行 | [查看](EXP-001-pipeline-smoke.md) |
| `EXP-002` | SASRec | 10k SASRec | 10,000 users | `completed` | embedding 可用于 RQ-VAE；推荐指标不完整 | [查看](EXP-002-10k-sasrec.md) |
| `EXP-003` | RQ-VAE | 10k baseline | 386,364 items | `failed` | codebook 严重塌缩，mapping 不可用于 GR | [查看](EXP-003-10k-rqvae-baseline.md) |
| `EXP-004` | RQ-VAE | Balanced RQ-VAE | 386,364 items | `completed` | 推理 collision rate 降至 2.89%，允许进入 GR smoke | [查看](EXP-004-balanced-rqvae.md) |
| `EXP-005` | Qwen2 GR | FP16 / BF16 smoke 对照 | 40 train samples | `completed` | 两种精度均跑通；BF16 无非有限梯度，作为后续默认方案 | [查看](EXP-005-gr-smoke-precision-comparison.md) |
| `EXP-006` | Qwen2 GR | Qwen2.5-0.5B 10k baseline | 10,000 test users | `completed` | 生成 100% 合法；HR@10 0.81%，目录覆盖率 0.338%，效果和多样性不足 | [查看](EXP-006-qwen25-05b-10k-gr-baseline.md) |
| `EXP-007` | GR Baseline | Random / MostPopular Top-10 | 10,000 test users | `completed` | Random HR@10 0.01%，MostPopular 0.73%；GR 仅小幅超过热门度基线 | [查看](EXP-007-random-most-popular-baselines.md) |
| `EXP-008` | Qwen2 GR | Qwen2.5-0.5B 10k，2 epochs | 10,000 test users | `completed` | HR@10 提升至 1.13%，覆盖率提升至 0.599%；延长训练有效但绝对效果仍低 | [查看](EXP-008-qwen25-05b-10k-2epoch.md) |
| `EXP-009` | Qwen2 GR | Qwen2.5-0.5B 10k，2 epochs，lr=5e-5 | 10,000 test users | `completed` | 训练稳定但 HR@10 为 0.98%，低于 `EXP-008` 的 1.13%；3e-5 当前更优 | [查看](EXP-009-qwen25-05b-10k-2epoch-lr5e5.md) |
| `EXP-010` | SASRec | SASRec2 temporal 10k baseline | 10,000 test users | `completed` | item-level HR@10 0.52%，NDCG@10 0.002697；补齐时间切分 SASRec 参照 | [查看](EXP-010-sasrec2-temporal-baseline.md) |
| `EXP-011` | Qwen2 GR | 从两轮最佳模型低学习率续训 1 epoch | 10,000 validation users | `completed` | validation HR@10 1.32%，NDCG@10 0.007582，SID 覆盖率 0.833%；test 尚未运行 | [查看](EXP-011-qwen25-05b-10k-ep3-from-best.md) |

## 3. 当前决策与未解决问题

### 决策

| ID | 日期 | 决策 | 原因 | 依据 | 状态 |
|---|---|---|---|---|---|
| `DEC-001` | 2026-09-22 | 暂不进入 GR | collision rate 为 99.08% | `EXP-003` | superseded by `DEC-003` |
| `DEC-002` | 2026-09-23 | 使用独立目录开展平衡版 RQ-VAE | 需要验证 codebook collapse 缓解方案 | `EXP-003` | completed |
| `DEC-003` | 2026-09-23 | 允许使用平衡版 mapping 进行 GR smoke | collision rate 为 2.89%，覆盖率为 100% | `EXP-004` | completed |
| `DEC-004` | 2026-09-23 | 后续 GR 实验默认使用 BF16 | BF16 的 20 个 grad norm 均有限；FP16 有 9/20 个为 NaN/Inf | `EXP-005` | active |
| `DEC-005` | 2026-09-23 | 使用 step 26,000 最佳模型进行测试集评估 | validation loss 在 13 次评估中持续下降，step 26,000 最低 | `EXP-006` | completed |
| `DEC-006` | 2026-09-23 | 后续优先改善推荐相关性与候选覆盖，而非生成合法性 | 合法率和 mapping 率均为 100%，但 HR@10 和目录覆盖率较低 | `EXP-006` | active |
| `DEC-007` | 2026-09-23 | 将 Random 和 MostPopular 作为后续 GR 实验的固定参照 | GR 的 HR@10 仅比 MostPopular 高 0.08 个百分点，需要持续衡量个性化收益 | `EXP-007` | active |
| `DEC-008` | 2026-09-24 | 继续比较更多 epoch 和学习率配置 | 两轮训练较单轮 HR@10 相对提升 39.51%，覆盖率相对提升 77.01%，延长训练方向有效 | `EXP-008` | active |
| `DEC-009` | 2026-09-24 | 相同 2-epoch 设置优先使用 learning rate `3e-5` | `5e-5` 训练稳定但 test HR@10、NDCG、MRR 和覆盖率均低于 `3e-5` | `EXP-009` | active |
| `DEC-010` | 2026-09-24 | SASRec2 与 GR 暂不直接比较原始 HR/NDCG | SASRec2 是 item-level，GR 是 Semantic-ID-level；指标命中定义不同 | `EXP-010` | active |
| `DEC-011` | 2026-09-25 | 完成 history50 和 beam validation 后再选择唯一 test 配置 | 续训 validation 指标继续改善，但需要区分历史长度和解码宽度的影响，避免对多个候选反复查看 test | `EXP-011` | active |

### 未解决问题

| 问题 | 关联实验 | 状态 |
|---|---|---|
| 平衡训练中各项配置对改善结果的独立贡献尚未确认 | `EXP-003`、`EXP-004` | open |
| 10k SASRec 尚未完成 HR@10/NDCG@10/Recall@10 | `EXP-002` | open |
| 续训 validation 覆盖 3,127 个 Semantic ID，目录覆盖率提升至 0.833%，但候选分布仍较集中 | `EXP-008`、`EXP-011` | open |
| 延长训练扩大了相对 MostPopular 的优势，但候选集中是否主要源于热门度偏置尚未确认 | `EXP-006`、`EXP-007`、`EXP-008`、`EXP-011` | open |
| 更高学习率在当前 2-epoch 设置下导致效果下降，最佳学习率与 epoch 的交互尚未确认 | `EXP-008`、`EXP-009` | open |
| 从两轮最佳模型低学习率续训提升了 validation 覆盖率，但训练历史长度、评估历史长度和 beam width 的独立影响尚未确认 | `EXP-008`、`EXP-011` | open |
| GR 尚未完成与 SASRec2 相同口径的 item-level 评估 | `EXP-006`、`EXP-008`、`EXP-009`、`EXP-010`、`EXP-011` | open |
| validation/test 中约 4.2%-4.3% 的目标存在 Semantic ID 歧义，item 级消歧与重排尚未实现 | `EXP-004`、`EXP-006`、`EXP-008`、`EXP-011` | open |

## 4. 关键产物

这里只登记跨阶段使用或具有重要历史价值的产物。完整日志和产物见对应实验文档。

| 产物 | 来源实验 | 路径 | 状态 |
|---|---|---|---|
| SASRec checkpoint | `EXP-002` | `outputs/10k/sasrec/checkpoints/global_step32.valid_loss=1.2027/` | 可用 |
| item embeddings | `EXP-002` | `outputs/10k/emb/embeddings.npz` | 可作为 RQ-VAE 输入 |
| RQ-VAE checkpoint | `EXP-003` | `outputs/10k/rqvae/checkpoints/09-22-2026/` | 失败对照 |
| Semantic ID mapping | `EXP-003` | `outputs/10k/emb_infer/sinkhorn/worker_0_output.txt` | 禁止用于 GR |
| Balanced RQ-VAE checkpoint | `EXP-004` | `outputs/10k_balanced/rqvae/checkpoints/09-23-2026/best_collision_model.pth` | 可用 |
| Balanced Semantic ID mapping | `EXP-004` | `outputs/10k_balanced/emb_infer/sinkhorn/worker_0_output.txt` | 可用于 GR smoke |
| Balanced experiment summary | `EXP-004` | `outputs/10k_balanced/experiment_summary.json` | 不可变结果快照 |
| GR FP16 smoke model | `EXP-005` | `outputs/gr/checkpoints/qwen25_05b_smoke/model.safetensors` | 工程对照；不建议续训 |
| GR BF16 smoke model | `EXP-005` | `outputs/gr/checkpoints/qwen25_05b_smoke_bf16/model.safetensors` | 数值稳定性通过 |
| Qwen2.5-0.5B 10k 最佳模型 | `EXP-006` | `outputs/gr/checkpoints/qwen25_05b_10k_v1/model.safetensors` | step 26,000 最佳权重，已评估 |
| Qwen2.5-0.5B 最佳 checkpoint | `EXP-006` | `outputs/gr/checkpoints/qwen25_05b_10k_v1/checkpoint-26000/` | validation loss 5.4284 |
| Qwen2.5-0.5B 最终训练状态 | `EXP-006` | `outputs/gr/checkpoints/qwen25_05b_10k_v1/checkpoint-27356/` | 1 epoch 完成 |
| Qwen2.5-0.5B 测试指标 | `EXP-006` | `outputs/gr/checkpoints/qwen25_05b_10k_v1/test_metrics.json` | 10k Semantic-ID 级 Top-10 指标 |
| Qwen2.5-0.5B 测试预测 | `EXP-006` | `outputs/gr/checkpoints/qwen25_05b_10k_v1/test_predictions.jsonl` | 10,000 用户预测明细 |
| Random / MostPopular 指标汇总 | `EXP-007` | `outputs/gr/baselines/10k_balanced/semantic_id_top10/comparison.json` | 10k Semantic-ID 级 Top-10 基线 |
| Random / MostPopular 预测 | `EXP-007` | `outputs/gr/baselines/10k_balanced/semantic_id_top10/` | 各 10,000 用户预测明细 |
| Qwen2.5-0.5B 两轮最佳模型 | `EXP-008` | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/model.safetensors` | step 54,000 最佳权重，已评估 |
| Qwen2.5-0.5B 两轮最终状态 | `EXP-008` | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/checkpoint-54712/` | 2 epochs 完成 |
| Qwen2.5-0.5B 两轮评估产物 | `EXP-008` | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr3e5_b32/` | validation/test 指标、预测与完整性检查 |
| Qwen2.5-0.5B 两轮高学习率最佳模型 | `EXP-009` | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr5e5_b32/model.safetensors` | step 54,000 最佳权重，已评估 |
| Qwen2.5-0.5B 两轮高学习率评估产物 | `EXP-009` | `outputs/gr/checkpoints/qwen25_05b_10k_ep2_lr5e5_b32/` | validation/test 指标、预测与完整性检查 |
| SASRec2 temporal 最佳模型 | `EXP-010` | `outputs/sasrec2/temporal_10k_mm81_82/checkpoints/best_model.pt` | epoch 3，validation loss 0.003217 |
| SASRec2 temporal 评估产物 | `EXP-010` | `outputs/sasrec2/temporal_10k_mm81_82/` | item-level validation/test 指标与预测 |
| Qwen2.5-0.5B 续训最佳模型 | `EXP-011` | `outputs/gr/checkpoints/qwen25_05b_10k_ep3_from_best_lr1e5_b32/model.safetensors` | step 26,000 最佳权重，validation 已评估 |
| Qwen2.5-0.5B 续训最终状态 | `EXP-011` | `outputs/gr/checkpoints/qwen25_05b_10k_ep3_from_best_lr1e5_b32/checkpoint-27356/` | 额外 1 epoch 完成 |
| Qwen2.5-0.5B 续训 validation 产物 | `EXP-011` | `outputs/gr/ep3_history50_pipeline/attempts/ep3_validation_maxlen100/attempt-1-20260925T004528/` | 10,000 用户指标与预测，独立完整性检查通过 |
