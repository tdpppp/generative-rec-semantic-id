# TencentGR-1M 实验记录与日志阅读指南

本文档记录实际运行结果，不把架构文档中的示例数字当成本次实验结果。每次开始新规模、
新特征组合或新超参数实验时，应在文末复制一份实验模板并填写。

## 1. 目前完成了什么

当前完成的是一个三阶段流水线中的前两阶段试运行：

```text
TencentGR-1M Parquet
  -> profile 缓存：筛选用户、对齐特征、建立局部 ID
  -> SASRec：学习商品的 32 维表示
  -> RQ-VAE：尝试把 32 维表示量化为三层 Semantic ID
  -> Qwen2 GR：尚未开始
```

`smoke` 用来验证代码链路；`10k` 用来观察真实训练行为。它们都来自 TencentGR-1M，
不是两套不同格式的数据。

## 2. 数据与 ID

| 名称 | 用途 | 是否跨阶段保留 |
|---|---|---|
| `local_ids` | profile 内部紧凑 ID，降低 SASRec embedding 表规模 | 否 |
| `ids` | 官方 re-indexed item ID，`seq.item_id` 使用它 | 是 |
| `original_ids` | 原始商品 ID，用于和 `anonymous_cid` 对齐 | 用于追溯 |

RQ-VAE 和 GR 必须使用 `ids`。不能把 `local_ids` 当成最终商品 ID。

## 3. 本次实际结果

### 3.1 Smoke

| 项目 | 结果 |
|---|---:|
| 用户 | 256 |
| 商品 | 19,829 |
| 81 有效/缺失 | 18,674 / 1,155 |
| 82 有效/缺失 | 19,603 / 226 |
| SASRec steps | 2 |
| train loss | 1.9430 -> 1.8679 |
| validation loss | 1.8313 |

结论：数据读取、GPU 前向/反向、验证、checkpoint 和向量导出均可工作。两步 loss 不能证明
模型已经收敛，只能证明链路正确。

### 3.2 10k SASRec

| 项目 | 结果 |
|---|---:|
| 用户 | 10,000 |
| 商品 | 386,364 |
| profile 缓存 | 858 MB |
| 81 有效/缺失 | 363,712 / 22,652 |
| 82 有效/缺失 | 381,468 / 4,896 |
| train steps | 32 |
| train loss | 2.0001 -> 1.2299 |
| 平均 train loss | 1.4404 |
| validation loss | 1.2027 |
| 导出向量 | 386,364 x 32, float32 |
| ID 唯一性、NaN/Inf | 全部唯一、全部有限 |

结论：SASRec loss 明显下降，Stage 1 工程链路正常。当前只有 BCE loss；尚未实现或产出
`HR@10`、`NDCG@10`、`Recall@10`，因此不能声称推荐效果已经得到正式验证。

### 3.3 10k RQ-VAE 首次试验

训练配置：

```text
epochs=20, batch_size=4096, warmup_epochs=50, eval_step=20
num_emb_list=[2048, 2048, 1024]
sk_epsilons=[0.0, 0.0, 0.0]
```

| 项目 | 结果 |
|---|---:|
| 输入商品 | 386,364 |
| reconstruction epoch sum | 3.4252 -> 1.0720 |
| 训练期碰撞率 | 99.51% |
| 推理后唯一 Semantic ID | 3,552 |
| 推理后碰撞商品 | 382,812 |
| 推理后碰撞率 | 99.08% |
| 第一层 code 使用 | 12 / 2,048 |
| 第二层 code 使用 | 20 / 2,048 |
| 第三层 code 使用 | 235 / 1,024 |

结论：重建能力在改善，但码本严重塌缩。当前 Semantic ID 不可用于 GR。主要问题是训练
只有 20 epoch，却设置了 50 epoch warmup；碰撞只在最后评估一次；训练阶段未启用
Sinkhorn 平衡。

## 4. 日志分别在哪里

```text
outputs/10k/sasrec/logs/train.log
    每个训练 step 一行 JSON，包含 step、loss、epoch 和时间

outputs/10k/sasrec/tensorboard/
    SASRec TensorBoard event

outputs/10k/sasrec/checkpoints/global_step32.valid_loss=1.2027/
    Stage 1 checkpoint；目录名包含验证 loss

outputs/10k/emb/embeddings.npz
    Stage 1 输出：ids/original_ids/local_ids/embs

outputs/10k/rqvae/tensorboard/
    RQ-VAE 的 step loss、epoch loss 和 collision rate

outputs/10k/rqvae/checkpoints/09-22-2026/
    RQ-VAE checkpoint

outputs/10k/emb_infer/sinkhorn/worker_0_output.txt
    item re-ID 到 Semantic ID 的映射
```

## 5. 怎样看 SASRec 日志

查看原始记录：

```bash
head outputs/10k/sasrec/logs/train.log
tail outputs/10k/sasrec/logs/train.log
```

单步 `loss` 是正样本 BCE 与负样本 BCE 之和。趋势下降说明模型逐渐拉高正样本分数、
压低负样本分数。它不是准确率，也不能替代 HR/NDCG。训练 loss 低而验证 loss 上升时，
通常表示过拟合或训练/验证分布不同。

## 6. 怎样看 TensorBoard

```bash
source .venv/bin/activate
tensorboard --logdir outputs/10k --host 0.0.0.0 --port 6006
```

浏览器访问 `http://服务器地址:6006`；无法直连时使用 SSH 端口转发。关注：

| 指标 | 含义 | 健康趋势 |
|---|---|---|
| `Loss/train` | SASRec batch BCE loss | 总体下降，允许抖动 |
| `Loss/valid` | SASRec 验证 BCE loss | 下降后稳定；持续上升需警惕 |
| `Step/loss_recon` | RQ-VAE 单 batch 重建误差 | 总体下降 |
| `Step/loss_total` | 重建误差 + 量化损失 | 不应 NaN；不一定单调 |
| `Epoch/collision_rate` | 多商品共享 Semantic ID 的比例 | 越低越好 |

当前 RQ-VAE 的 `Epoch/loss_train` 和 `Epoch/loss_recon` 是一个 epoch 内 batch loss 的
总和，不是均值；只适合在 batch 数相同的 epoch 间比较趋势。

## 7. 各指标的通过标准

| 指标 | Smoke | 可进入 GR 的建议标准 |
|---|---:|---:|
| embedding NaN/Inf | 必须为 0 | 必须为 0 |
| item ID 覆盖率 | 100% | 100% |
| item ID 唯一率 | 100% | 100% |
| Semantic ID 碰撞率 | 只观察 | `<10%`，最好 `<5%` |
| RQ-VAE codebook usage | 只观察 | 不应只有个位数/十位数 code |
| HR/NDCG/Recall | 可暂缺 | 正式结论前必须提供 |

Loss 只能和同数据、同切分、同负采样、同模型配置的实验比较。不能跨任务直接比较
SASRec BCE loss 与 RQ-VAE MSE。

## 8. 自动汇总

```bash
python tools/summarize_experiment.py --run_dir outputs/10k
```

保存一份不可变结果快照：

```bash
python tools/summarize_experiment.py \
  --run_dir outputs/10k \
  --output outputs/10k/experiment_summary.json
```

脚本会汇总 SASRec train/validation loss、NPZ shape 和有限性、RQ-VAE TensorBoard 标量、
Semantic ID 覆盖率、碰撞率和各层 code 使用量。

## 9. 下一次实验模板

```text
实验名称：
日期和 Git commit：
目标/假设：
数据 profile：
用户数/商品数：
特征组合：
随机种子：
完整命令：
输出目录：

SASRec：
  steps/epochs：
  first/last/min train loss：
  validation loss：
  HR@10/NDCG@10/Recall@10：
  embedding shape/NaN/Inf：

RQ-VAE：
  epochs/warmup/eval_step：
  codebook sizes：
  reconstruction loss：
  各层 code 使用量：
  collision rate：

结论：
是否通过质量门槛：
失败原因或下一步：
```

## 10. 当前下一步

保留首次 RQ-VAE 结果作为失败对照，使用独立目录重新训练平衡版 RQ-VAE。不要用当前
99.08% 碰撞率的映射训练 Qwen2。新实验至少每 10 epoch 记录碰撞率，并在碰撞率低于
10% 后再进入 GR smoke。
