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

### 运行流程说明：

1. **环境检查**: 从仓库根目录运行 `python environment_check.py`。
2. **路径解析**: `USER_CACHE_PATH` 提供模型、数据与 Semantic ID 映射的根目录。
3. **启动训练**: `run_train.sh` 从仓库根目录调用 `python -m gr.train_gr` 并读取 `gr_train.json`。

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
