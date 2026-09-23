# LLM-Enhanced Generative Recommender with Semantic ID

基于多模态商品表示、Residual Quantization 和 Qwen2 的三阶段生成式推荐项目。

## Pipeline

```text
TencentGR user sequences + item features
                  |
                  v
Stage 1: SASRec -> outputs/emb/embeddings.npz
                  |
                  v
Stage 2: RQ-VAE -> item_id -> <a_x><b_y><c_z>
                  |
                  v
Stage 3: Qwen2 SFT -> next-item semantic-ID generation
```

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)，面试速成路线见
[LEARNING_PLAN.md](LEARNING_PLAN.md)，高频问答见 [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)，
真实运行结果和日志阅读方法见 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

## Environment

推荐 Python 3.10、PyTorch 2.1+ 和 CUDA GPU。先创建独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python environment_check.py --strict
python -m unittest discover -s tests -v
```

如果系统提示缺少 `ensurepip`，可以改用：

```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
```

`environment_check.py` 会分别报告 Git、NVIDIA 驱动、Python 包、本地数据、GR 配置、
Qwen2 权重、训练序列和 Semantic ID 映射；不加 `--strict` 时只报告问题而不会返回
失败状态。

## Stage 1: SASRec

训练和 smoke test 都直接使用官方 `data/TencentGR_1M` Parquet 数据。先检查下载内容：

```bash
python tools/inspect_tencentgr.py --data_path data/TencentGR_1M
```

执行 smoke profile。第一次运行会从 1M 数据中确定性选择 256 个用户、构建诱导商品集合，
并扫描 81/82 分片生成可复用缓存；之后会直接复用缓存：

```bash
python sasrec/main.py \
  --data_path data/TencentGR_1M \
  --profile smoke \
  --mm_emb_id 81 82 \
  --max_steps 20 \
  --num_epochs 1 \
  --batch_size 128 \
  --ckpt_dir outputs/sasrec/checkpoints \
  --log_dir outputs/sasrec/logs \
  --tensorboard_dir outputs/sasrec/tensorboard \
  --emb_output_dir outputs/emb
```

`--profile` 支持 `smoke`、`10k`、`100k` 和 `full`，四种规模使用相同的 Parquet
解析与缓存代码。也可以通过 `--max_users` 覆盖用户数。缓存位于
`data/TencentGR_1M/cache/profiles`，删除缓存或传 `--rebuild_cache` 才会重新扫描原始分片。
也可以在训练前显式构建，例如：

```bash
python tools/build_tencentgr_profile.py \
  --data_path data/TencentGR_1M \
  --profile 10k \
  --mm_emb_id 81 82
```

导出的 `embeddings.npz` 中，`ids` 是官方序列使用的 re-indexed item ID；
`original_ids` 和 `local_ids` 用于追溯，不能用局部 ID 代替 `ids` 训练后续阶段。

## Stage 2: RQ-VAE

训练语义码本：

```bash
python rqvae/train/rqvae_train.py \
  --data_path outputs/emb \
  --ckpt_dir outputs/rqvae/checkpoints \
  --log_dir outputs/rqvae/tensorboard \
  --epochs 1000 \
  --batch_size 8192
```

生成 Semantic ID。`--checkpoint` 应指向训练输出的 `best_collision_model.pth`：

```bash
python rqvae/infer/rqvae_infer.py \
  --data_path outputs/emb \
  --checkpoint outputs/rqvae/checkpoints/DATE/best_collision_model.pth \
  --output_dir outputs/emb_infer/sinkhorn
```

映射文件输出为 `outputs/emb_infer/sinkhorn/worker_0_output.txt`。

## Stage 3: Qwen2 SFT

准备以下文件：

```text
outputs/qwen_init2/                         Qwen2 config/tokenizer/weights
data/TencentGR_1M/seq/*.snappy.parquet     官方用户行为序列
outputs/emb_infer/sinkhorn/worker_0_output.txt
```

然后将 `gr/gr_train.json` 中的 `item2token_dict` 改为 `emb_infer/sinkhorn`，运行：

```bash
export USER_CACHE_PATH="$PWD/outputs"
export TRAIN_CKPT_PATH="$PWD/outputs/gr/checkpoints"
bash gr/run.sh
```

GR 会直接流式读取 Parquet，为 `<a_*>`、`<b_*>`、`<c_*>` 和历史边界标记注册独立
token，并且只对目标商品的 Semantic ID 计算语言模型损失。做 smoke test 时可在
`gr/gr_train.json` 设置 `max_train_samples`，正式训练设为 `null`。

## Repository Checks

以下检查不启动长时间训练：

```bash
python environment_check.py
python -m unittest discover -s tests -v
python -m compileall -q -f .
```

训练产物、数据、checkpoint 和本地虚拟环境已由 `.gitignore` 排除；训练配置
`gr/gr_train.json` 应当纳入 Git 版本控制。

## References

- TIGER: Recommender Systems with Generative Retrieval, arXiv:2305.05065
- Microsoft RecAI
