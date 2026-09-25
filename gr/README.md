# Generative Recommendation SFT Training Pipeline

这是一个基于 **Qwen2** 模型进行微调（SFT）的生成式推荐（Generative Recommendation）训练项目。该项目旨在通过将用户历史行为序列化为 Semantic ID 序列，训练 LLM 预测下一个感兴趣的 Item。

## 📁 项目结构

```
.
├── run.sh                 # [入口脚本] 环境安装与启动训练
├── run_train.sh           # [启动脚本] 配置分布式环境变量 (Gloo/TCP) 并启动 Python 脚本
├── train_gr.py            # [主程序] 训练入口，负责模型加载、Trainer 初始化
├── gr_train.json          # [配置文件] 模型参数、数据参数、训练参数配置
├── arguments.py           # [参数定义] 定义 dataclasses (Model/Data/Training Arguments)
├── custom_dataset.py      # [数据处理] 流式读取 Parquet/JSON 并构造模型输入
├── utils.py               # [工具类] 模型加载 (Qwen2)、Token 字典加载等
└── requirements.txt       # Python 依赖列表
```

## 🛠️ 环境依赖

项目主要依赖 **PyTorch**, **Transformers**, **DeepSpeed** 和 **Accelerate**。

先在仓库根目录执行 `python -m pip install -r requirements.txt` 安装依赖。`run.sh` 只负责
启动训练，不会在训练任务中修改 Python 或系统环境。主要依赖包含：

- Python 3.8+
- PyTorch (CUDA 11.7 / 12.x)
- DeepSpeed == 0.15.4
- Transformers == 4.46.3
- HuggingFace Hub == 0.33.4
- MPI (OpenMPI)

## 📊 数据准备

在运行之前，你需要准备好模型权重文件和训练数据，并确保存储在环境变量指定的路径下。

### 1. 环境变量设置

代码依赖以下环境变量来寻找数据和保存模型：

- `USER_CACHE_PATH`: 存放预训练模型、训练数据、字典文件的根目录。
- `TRAIN_CKPT_PATH`: 存放训练输出（Checkpoints）的目录。
- `RUNTIME_SCRIPT_DIR`: 当前脚本所在目录（通常由平台自动设置，或需手动指定）。

### 2. 模型文件 (Qwen2 Init)

请确保在 `$USER_CACHE_PATH/qwen_init2` 路径下包含 Qwen2 的初始化权重和配置文件（`config.json`, `tokenizer.json` 等）。

### 3. Token 映射字典 (Item2Token)

推荐系统使用 Semantic ID，需要提供 Item 到 Token 的映射文件。

- **路径**: `$USER_CACHE_PATH/emb_infer/sinkhorn10` (可在 `gr_train.json` 中修改 `item2token_dict`)
- **格式**: 文件夹，内部包含文本文件。
- **内容格式**: 每一行 `item_id \t token` (Tab 分隔)。

### 4. 训练数据格式

默认直接流式读取 `data/TencentGR_1M/seq/*.snappy.parquet`。每行包含 `user_id` 和
由 `item_id/action_type/timestamp` 组成的 `seq`。Semantic ID 映射中的 key 必须使用
同一套官方 re-indexed item ID。`data_format` 设为 `json` 时仍可读取旧 JSON，但正式
链路不再依赖它。序列长度小于 2 的用户会被忽略；`max_train_samples` 可限制 smoke 规模。

## 🚀 运行训练

该项目设计为通过 `run.sh` 一键启动。

Bash

```
bash run.sh
```

### 10k Qwen2.5-0.5B 正式基线

正式基线直接读取 10k profile 缓存。缓存内的 local item ID 会通过 `item_reids.npy`
还原为官方 re-ID，再与通过质量检查的 Semantic ID mapping 对齐。时间切分规则为：训练集
使用倒数两个商品之前的滑动前缀，验证集预测倒数第二个商品，测试集预测最后一个商品。

```bash
export USER_CACHE_PATH="$PWD/outputs"
export TRAIN_CKPT_PATH="$PWD/outputs/gr/checkpoints"
python -m gr.train_gr --config gr/gr_train_qwen25_05b_10k_v1.json
```

当前 10k profile 产生 875,365 个训练样本、10,000 个验证样本和 10,000 个测试样本。
正式配置使用 Qwen2.5-0.5B Base、BF16、batch size 32，并根据 validation loss 保留最佳
checkpoint。

训练结束后进行受约束的 Top-10 测试。评估器通过 Semantic ID 前缀树保证输出顺序为
`a -> b -> c`，并且三层组合必须存在于 mapping：

```bash
python -m gr.evaluate_gr \
  --model_path outputs/gr/checkpoints/qwen25_05b_10k_v1 \
  --profile_path data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82 \
  --mapping_dir outputs/10k_balanced/emb_infer/sinkhorn \
  --split test \
  --batch_size 16 \
  --top_k 10 \
  --device cuda:0 \
  --output outputs/gr/checkpoints/qwen25_05b_10k_v1/test_metrics.json \
  --predictions outputs/gr/checkpoints/qwen25_05b_10k_v1/test_predictions.jsonl
```

输出包括 Semantic-ID 级 `HR@10`、`Recall@10`、`NDCG@10`、`MRR@10`、合法格式率、
合法 mapping 率和目录覆盖率。存在碰撞的 Semantic ID 仍需要单独的 item 级重排策略，不能
把 Semantic-ID 命中直接表述为无歧义的 item 命中。

### GR 非神经基线评估

使用与模型评估相同的 test split 运行 Random 和 MostPopular Top-10 基线。MostPopular
只统计每个序列中 validation、test 留出项之前的事件，避免把评估目标泄漏到热门度统计中：

```bash
python -m gr.evaluate_gr_baselines \
  --profile_path data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82 \
  --mapping_dir outputs/10k_balanced/emb_infer/sinkhorn \
  --split test \
  --top_k 10 \
  --seed 2025 \
  --output_dir outputs/gr/baselines/10k_balanced/semantic_id_top10
```

结果目录包含两个基线各自的指标 JSON 和逐用户预测 JSONL，以及便于直接比较的
`comparison.json`。Random 基线固定随机种子；两个基线都只生成 mapping 中存在且互不重复的
Semantic ID。

### 运行流程说明：

1. **环境检查**: 从仓库根目录运行 `python environment_check.py`。
2. **路径解析**: `USER_CACHE_PATH` 提供模型、数据与 Semantic ID 映射的根目录。
3. **启动训练**: `run_train.sh` 从仓库根目录调用 `python -m gr.train_gr` 并读取 `gr_train.json`。

GR checkpoint 默认写入 `outputs/gr/checkpoints/<output_dir>/`，TensorBoard event 默认写入
`outputs/gr/tensorboard/<output_dir>/`。可分别通过 `TRAIN_CKPT_PATH` 和
`TRAIN_TF_EVENTS_PATH` 修改两个根目录。

## ⚙️ 参数配置 (`gr_train.json`)

主要的超参数在 JSON 文件中修改：

| **参数模块**      | **关键参数**                  | **说明**                                     |
| ----------------- | ----------------------------- | -------------------------------------------- |
| **model_args**    | `se_id_space_width`           | 语义 ID 的空间宽度配置 (如 "2048,2048,1024") |
| **data_args**     | `max_seq_length`              | 用户行为序列的最大长度 (默认 100)            |
|                   | `token_depth`                 | Semantic ID 的层级深度 (默认 3)              |
| **training_args** | `output_dir`                  | 输出目录名 (位于 `$TRAIN_CKPT_PATH` 下)      |
|                   | `learning_rate`               | 学习率 (默认 1e-4)                           |
|                   | `per_device_train_batch_size` | 单卡 Batch Size                              |
|                   | `gradient_accumulation_steps` | 梯度累积步数                                 |

## 🧩 关键代码逻辑说明

- Prompt 构造 (custom_dataset.py):

  模型输入构造如下：

  Plaintext

  ```
  Input: <|hist_clk_start|> [Token_Item_1] [Token_Item_2] ... <|hist_clk_end|> [Target_Item_Token]
  Label: [Target_Item_Token] (仅对 Target 部分计算 Loss)
  ```

- **DeepSpeed**: 虽然 `run_train.sh` 主要是手动设置环境变量，但代码中预留了 DeepSpeed 的集成逻辑（通过 `transformers.Trainer` 支持）。

## ⚠️ 故障排除

- **网络连接问题**: 如果遇到 NCCL 超时或通信错误，请检查 `run_train.sh` 中的 `NCCL_SOCKET_IFNAME=lo`。默认配置绑定了本地回环接口用于调试，**在多机训练时必须修改为实际网卡名称 (如 `eth0`)**。
- **路径错误**: 确保 `USER_CACHE_PATH` 环境变量已正确导出，否则会报 `FileNotFoundError`。
