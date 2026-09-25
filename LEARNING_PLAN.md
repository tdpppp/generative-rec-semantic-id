# 生成式推荐项目：面试导向的阶段式学习计划

这份计划面向第一次接触生成式推荐、但需要围绕本项目准备推荐算法、生成式推荐、LLM
应用或算法工程面试的学习者。学习不再按“第几天”推进，而是按能力阶段推进：完成当前阶段
的通关标准后，再进入下一阶段。

建议总投入约 35-55 小时。时间充裕时完整完成所有任务；时间有限时优先完成标有
“必须”的内容，但不要跳过阶段 0、阶段 4 的离线评估和阶段 5 的模拟面试。

配套资料：

- [README.md](README.md)：环境、运行入口和三阶段命令。
- [ARCHITECTURE.md](ARCHITECTURE.md)：架构图和模块关系。
- [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)：高频追问与回答边界。

## 个人学习进度

更新时间：2026-09-25。

| 阶段 | 状态 | 当前掌握情况 |
|---|---|---|
| 阶段 0：项目地图与推荐基础 | 基础内容已完成 | 能说明三阶段职责、Semantic ID 的价值与局限，并区分中间指标和最终推荐指标 |
| 阶段 1：SASRec 与多模态商品表示 | 基础内容已完成 | 能解释 `seq/pos/neg`、负采样、三种 mask、位置编码、特征融合、点积打分和 BCE |
| 阶段 2：RQ-VAE 与 Semantic ID | 学习中 | 已掌握两层残差量化、三类损失及其梯度路径和 straight-through estimator；已进入码本初始化、死码、利用率与 Sinkhorn |
| 阶段 3：Qwen2 生成式推荐 | 未开始 | 等待阶段 2 通关后进入 |
| 阶段 4：离线评估与工程排障 | 未系统学习 | 已接触 Recall、NDCG、Coverage 和中间指标边界 |
| 阶段 5：面试表达与完整模拟 | 持续积累 | 已完成第一版 2 分钟项目介绍，后续随技术学习继续修订 |

状态表示当前学习位置，不代表相关内容已经通过代码实验完整验证。阶段 0 和阶段 1 仍需在
后续复习中巩固容易混淆的概念，并最终通过闭卷口述和代码阅读验收。

阶段 2 当前验收点：完成码本初始化、死码、码本利用率和 Sinkhorn 的四道练习，并能说明
K-Means 初始化与 Sinkhorn 的作用、代价和评估边界。通过后继续学习 Semantic ID 碰撞、
碰撞处理策略及其对下游推荐指标的影响。

## 学习方法与完成标准

每个主题都用同一套闭环学习，避免停留在“看懂了”：

1. **讲原理**：不用代码，说明它解决什么问题、为什么这样设计、有什么代价。
2. **走代码**：指出入口、核心函数、输入输出 shape、损失和产物。
3. **做验证**：运行检查、手算例子或最小实验，并保存可以举证的结果。
4. **答追问**：先给结论，再给机制、证据、局限和改进方案。

每完成一个阶段，至少留下三种产物：一张图或一页笔记、一段可运行或可手算的验证、一次
闭卷口述。不要以“文件读完了”作为完成标准。

### 知识优先级

**必须掌握**：SASRec、隐式反馈与负采样、因果注意力、RQ-VAE、Semantic ID、碰撞率、
Qwen2 Causal LM、label mask、Recall/NDCG、训练与推理数据流。

**能够解释**：Sinkhorn、K-Means 初始化、mixed precision、gradient accumulation、
`IterableDataset` 分片、checkpoint 恢复、constrained decoding。

**了解边界即可**：从头推导 Transformer 全部公式、DeepSpeed 内部实现、多机通信细节、
完整在线 Serving。被问到时说明当前项目聚焦离线多阶段训练链路，再给出演进方案。

---

## 阶段 0：建立项目地图与推荐基础

### 阶段目标

- 能在 2 分钟内说明业务问题、三阶段方案和每一阶段的作用。
- 分清传统“召回 + 排序”和生成式推荐的输出方式。
- 分清原始 item ID、re-id、码本 index、Semantic ID token，避免后续概念混乱。
- 明确哪些结果来自本地验证，哪些只是文档中的历史记录。

### 0.1 推荐系统的基本问题

先掌握以下概念：

- 隐式反馈：点击、收藏、购买只能表示观察到的正反馈，未点击不一定是不喜欢。
- next-item prediction：用按时间排列的历史行为预测下一次交互。
- 召回负责从大商品库取候选，排序负责对候选精细打分。
- 传统召回通常对候选打分或做向量近邻搜索；生成式推荐直接生成商品的离散标识。
- 数据必须按时间切分，不能让未来行为进入训练历史。

必须会解释三个离线指标：

```text
Recall@K  = Top-K 中命中的相关商品数 / 相关商品总数
HitRate@K = 至少命中一个目标的用户比例
NDCG@K    = DCG@K / IDCG@K，其中更靠前的命中获得更大权重
```

当每个用户只有一个测试目标时，单用户的 Recall@K 和 HitRate@K 数值相同；但二者的定义
并不相同。面试时要先说明评估协议，再解释指标。

### 0.2 项目全景

闭眼能画出下面的数据流：

```text
用户序列 + 商品特征
        |
        v
SASRec：序列监督 + 多模态融合
        |  embeddings.npz: item -> 32d vector
        v
RQ-VAE：连续向量残差量化
        |  worker_0_output.txt: item -> <a_x><b_y><c_z>
        v
Qwen2：根据历史 Semantic IDs 自回归预测下一商品
        |  generated code -> item mapping
        v
推荐候选与 Recall/NDCG/Coverage
```

阅读顺序：

1. 根目录 `README.md` 的 Pipeline 和三个运行入口。
2. `ARCHITECTURE.md` 的功能层与数据流图。
3. `INTERVIEW_GUIDE.md` 的“一页数据流”和“不能过度声称的内容”。
4. 运行 `source .venv/bin/activate && python environment_check.py`；没有环境时先按
   `README.md` 创建，不要求此时启动长时间训练。

### 0.3 面试表达边界

- 可以说“实现了三阶段生成式推荐原型或多阶段流水线”，不要把依次训练说成严格意义的
  端到端联合训练。
- 可以说“内容特征缓解新商品冷启动”，不要说“彻底解决冷启动”。
- `ARCHITECTURE.md` 中的历史 loss、碰撞率和死码率不能冒充本轮实验结果。
- 尚未完成的线上 Serving、A/B 实验和业务收益，要明确列为局限或下一步。

### 动手任务

1. 手算两个用户的 Recall@3 和 NDCG@3。
2. 给数据流的每条箭头补上文件格式、shape 和生产/消费模块。
3. 录一段 2 分钟项目介绍，结构为“问题 -> 方案 -> 三阶段 -> 验证 -> 局限”。

### 高频追问

- 为什么不用普通 item ID，或为每个商品增加一个 LLM token？
- 为什么不直接让 Qwen2 读取原始多模态向量？
- 生成式推荐怎样与传统召回做公平比较？
- 这个项目为什么不能直接声称已经取得业务收益？

### 通关标准

- 不看资料画出三阶段数据流，并说清每阶段的输入、模型、损失、输出和指标。
- 能在 2 分钟内讲完项目，不把中间指标当最终推荐指标。
- 能准确指出项目当前的实验与上线边界。

---

## 阶段 1：SASRec 与多模态商品表示

### 阶段目标

- 理解序列样本如何构造、负样本从哪里来、如何避免标签泄漏。
- 能从输入张量一路讲到正负样本打分和 `embeddings.npz`。
- 能解释 causal mask、BCE、padding mask 和多模态融合的作用。

### 1.1 数据构造与负采样

重点阅读 `tools/inspect_tencentgr.py`、`sasrec/tencentgr_dataset.py` 的 Parquet profile、
样本构造与 `collate_fn`。

必须掌握：

- `seq[t]` 用来预测 `pos[t]`，`neg[t]` 是用户未交互商品；`0` 是 padding。
- padding 位置不应参与有效训练，序列切分必须保持时间顺序。
- 随机负采样简单高效，但“未交互”不等于“不喜欢”，可能采到假负例。
- 改进方法包括曝光负样本、热门度校正、in-batch negatives、hard negatives 和
  sampled softmax；要能说明各自代价。

动手：用一个长度为 5 的商品序列手画 `seq/pos/neg`，再标出 padding 和有效 loss 位置。

### 1.2 模型与因果注意力

重点阅读 `sasrec/model.py` 的 `feat2emb`、`log2feats`、`forward` 和
`save_item_emb`。

沿下面的顺序解释代码：

```text
item/feature inputs
 -> item embedding + position embedding + projected multimodal features
 -> causal self-attention blocks
 -> sequence representation
 -> positive/negative logits
 -> BCEWithLogitsLoss
```

必须回答：

- causal mask 使位置 `t` 只能关注 `<= t`，否则训练时会偷看未来标签。
- 当前实现做逐位置正负样本二分类，所以使用 BCE，而不是对全商品做 softmax。
- 多模态特征投影到统一空间后融合，使商品表示同时承载内容和序列行为信号。
- 内容特征有助于新商品表示，但不能解决新用户、缺失内容和分布漂移。

### 1.3 导出商品向量

跟踪 `save_item_emb` 如何产生 `ids` 和 `embs`，确认：

- `embeddings.npz` 是阶段 1 和阶段 2 的文件契约。
- 当前文档约定商品向量维度为 32，但面试时应以实际配置和产物 shape 为准。
- 导出的商品向量与训练时的 ID 空间、特征版本、checkpoint 必须配套。

### 动手任务

1. 手写 scaled dot-product attention，并加 causal mask。
2. 打印一个 batch 的 `seq/pos/neg` shape，确认有效位置和 ID 范围。
3. 在资源允许时执行一轮最小训练；否则至少完成单 batch 前向并检查有限 loss。
4. 检查导出的 NPZ keys、shape、dtype、NaN/Inf 和 item 数量。

### 高频追问

- 为什么需要 causal mask？padding mask 和 causal mask 有什么区别？
- 为什么用 BCE，不用全量 softmax？
- 随机负采样会造成什么偏差？
- 为什么不直接量化原始多模态向量？这个判断如何用消融实验验证？

### 通关标准

- 能用一个具体序列解释训练样本和 loss 位置。
- 能结合代码讲清 `feat2emb -> log2feats -> forward -> save_item_emb`。
- 能提出“原始内容向量、纯 ID SASRec、多模态 SASRec”三组公平消融。

---

## 阶段 2：RQ-VAE 与 Semantic ID

### 阶段目标

- 理解为什么要把连续商品向量变成多层离散 code。
- 能手算两层残差量化并解释重建、codebook、commitment loss。
- 能分析碰撞、死码、码本容量和语义解释的边界。

### 2.1 从连续向量到残差量化

重点阅读 `rqvae/train/rqvae_model.py` 的 `VectorQuantizer`、
`ResidualVectorQuantizer` 和 `RQVAE`。

必须掌握：

```text
x -> encoder -> z
r0 = z
第 l 层：找到离 r_l 最近的码字 q_l，令 r_(l+1) = r_l - q_l
q = q_1 + q_2 + ... + q_L
q -> decoder -> x_hat
```

- 普通 VQ 用一个码字近似完整 latent；RQ-VAE 用后续码本继续逼近残差。
- 重建损失使离散表示保留输入信息；codebook/commitment 项让码字和编码器输出相互靠近。
- nearest-neighbor 选择不可导，straight-through estimator 让重建梯度近似传回编码器。
- 多层 code 提供很大的组合空间，但组合空间大不等于所有组合都有效或碰撞自然为零。

### 2.2 码本训练与利用率

重点阅读 `rqvae/train/trainer.py`，理解：

- K-Means 初始化让初始码字更贴近数据分布。
- 死码是长期没有样本分配的码字，可通过初始化、EMA/重置、平衡分配等缓解。
- Sinkhorn 近似平衡批内分配，提高利用率；强行平衡也可能损伤真实的非均匀语义结构。
- 需要同时观察重建误差、码本使用率/perplexity、碰撞率和下游推荐指标。

### 2.3 Semantic ID 与碰撞

重点阅读 `rqvae/infer/rqvae_infer.py`，掌握：

- 三层 code `[42, 128, 7]` 表示为 `<a_42><b_128><c_7>`。
- 可定义碰撞率为 `(商品数 - 唯一完整 code 数) / 商品数`；还应看最大碰撞组和分布。
- 碰撞导致一个生成 code 对应多个商品，必须定义映射与回退策略。
- 大码本通常降低碰撞，但会增加参数、稀疏更新和死码风险。
- 层次前缀表示由粗到细的残差信息，但不能未经验证就声称每层对应业务类目。

### 动手任务

1. 用二维向量和每层两个码字手算两层残差量化。
2. 手写最近邻量化和两层残差量化伪代码。
3. 写出按完整 code 分组、计算碰撞率和最大碰撞组的逻辑。
4. 复述已有 smoke test 的准确边界：64 个随机向量、小码本 `[8,8,4]`、两轮碰撞
   处理、64 行合法映射。这是工程验证，不是业务效果实验。

### 高频追问

- RQ-VAE 相比普通 VQ 的优势和代价是什么？
- straight-through estimator 为什么有用？
- 为什么有死码？Sinkhorn 是否一定越强越好？
- 降低重建 MSE 是否一定提高 Recall@K？

### 通关标准

- 能在白板上画出 encoder、三层残差量化和 decoder，并说明每项 loss。
- 能准确计算和解释碰撞率，不把码本组合数等同于有效商品数。
- 能讲清 `embeddings.npz -> checkpoint -> worker_0_output.txt` 的数据契约。

---

## 阶段 3：Qwen2 生成式推荐

### 阶段目标

- 理解训练样本、tokenizer 扩展、目标标签掩码和 Causal LM 训练目标。
- 能说明预训练 Qwen2 的收益为何需要实验验证。
- 能设计从 Semantic ID 生成到商品候选的合法推理流程。

### 3.1 训练样本与 label mask

重点阅读 `gr/custom_dataset.py`。

```text
input  = <hist_start> 历史商品 Semantic IDs <hist_end> 目标商品 Semantic ID
labels = -100 ... -100                           目标的各层 token
```

必须掌握：

- `-100` 被交叉熵忽略，使梯度集中在“根据历史预测目标”，而不是复述历史。
- 目标必须是最后一个可映射商品，历史只能来自目标之前，否则会标签泄漏。
- 序列截断后仍需保证目标 token 保留、历史边界完整、input 与 labels 对齐。
- `IterableDataset` 在多 worker/多 rank 下必须分片，否则样本会重复消费。

### 3.2 Tokenizer、模型和训练

重点阅读 `gr/utils.py` 和 `gr/train_gr.py`。

- `from_pretrained` 才加载预训练参数；只用 config 构造模型是随机初始化。
- `<a_42>` 等 Semantic ID 应是独立 token，不能被拆成普通字符片段。
- 扩词表后必须 `resize_token_embeddings`。
- tokenizer、模型 checkpoint、Semantic ID 深度与每层空间宽度必须版本一致。
- fp16 可以降低显存、提高吞吐，但需要注意数值溢出。
- 梯度累积增大等效 batch，不会减少总计算量；等效 batch 还与设备数相关。

关于“为什么选 Qwen2”，面试中的稳健回答是：它有成熟的 Hugging Face Causal LM 接口
和中文预训练能力，工程接入成本较低；但纯 Semantic ID 与自然语言分布差异很大，预训练
是否有效必须与同规模随机初始化 Transformer 做消融。

### 3.3 推理与合法解码

设计完整推理链路：

```text
用户历史 item IDs
 -> 查表得到历史 Semantic IDs
 -> Qwen2 自回归生成 <a_*>、<b_*>、<c_*>
 -> 校验/查找完整 code
 -> 处理未见 code、碰撞、重复候选
 -> 商品候选列表
```

constrained decoding 应限制第一位只能选 `<a_*>`、第二位只能选 `<b_*>`、第三位只能选
`<c_*>`；进一步可用 trie 只允许生成映射表中真实存在的完整 code。Beam search 能提供多个
候选，但增加计算成本，也仍需去重和碰撞处理。

### 动手任务

1. 构造一个最小样本，逐 token 写出 `input_ids` 和 labels 中的 `-100`。
2. 检查一个 Semantic ID 经 tokenizer 编码后，每层是否正好对应一个 token。
3. 手写“历史 labels 置为 `-100`”的数据处理伪代码。
4. 画出带 constrained decoding、反向映射和异常回退的推理流程。

### 高频追问

- 为什么只对目标 token 计算 loss？全序列 loss 会怎样？
- 为什么 Semantic ID 必须注册成独立 token？
- 为什么使用 Qwen2，而不是从零训练小 Transformer？
- 怎样防止非法 code？碰撞 code 和未见 code 怎样处理？

### 通关标准

- 能从一条用户序列完整推导训练输入和 labels。
- 能结合代码说明权重加载、扩词表、resize 和训练器初始化。
- 能设计一个不会悄悄吞掉非法结果的推理回退策略。

---

## 阶段 4：离线评估、实验设计与工程排障

### 阶段目标

- 把“模型能训练”与“推荐有效”严格区分。
- 能设计公平基线、消融实验、分桶指标和可复现实验记录。
- 面对 OOM、NaN、loss 下降但指标不涨时，有固定排查顺序。

### 4.1 离线评估协议

建议按时间为每个用户留出最后一次或最后若干次交互，统一候选集合和过滤规则，然后报告：

- Recall@K、HitRate@K、NDCG@K。
- Coverage、合法 code 比例、未见 code 比例、重复候选比例。
- 热门/长尾/新商品、短历史/长历史等分桶结果。
- 推理延迟和资源消耗，避免只比较效果不比较代价。

至少准备这些对照：热门推荐、ID-based SASRec、直接量化内容向量、完整方案。消融实验一次
只改变一个因素，固定数据切分、候选集合、随机种子、评估代码和训练预算；条件允许时报告
多个 seed 的均值与波动。

### 4.2 阶段指标与最终指标

| 阶段 | 健康度指标 | 最终要回答的问题 |
|---|---|---|
| SASRec | train/valid loss、向量数值与覆盖 | 是否提升 Recall/NDCG |
| RQ-VAE | 重建误差、使用率、碰撞率 | 离散化是否保留推荐能力 |
| Qwen2 | target-token loss、合法率 | 生成候选是否命中且排序靠前 |

中间 loss 下降不能替代最终 Recall/NDCG。一个重建更好的码本也可能没有保留对下一商品预测
最重要的结构。

### 4.3 固定排障顺序

1. 检查路径、配置、tokenizer、映射和 checkpoint 是否属于同一实验。
2. 检查样本数、ID 范围、dtype、shape、padding、NaN/Inf。
3. 检查单 batch 前向、各分项 loss 是否有限。
4. 检查反向传播、非零梯度、梯度范数和参数是否更新。
5. 尝试在极小数据集上过拟合。
6. 再逐步扩大数据、模型、batch 和分布式规模。

准备三个故障案例：CUDA OOM、loss NaN、loss 下降但 Recall 不涨。每个案例按照“现象 ->
定位 -> 根因 -> 修复 -> 验证”回答，不要只罗列可能原因。

### 4.4 工程契约与可复现性

最大的工程风险之一是阶段间契约漂移。至少记录：

- Git commit、随机种子、数据版本与时间切分。
- SASRec checkpoint 和导出向量的维度、ID 空间、特征版本。
- RQ-VAE checkpoint、码本宽度、深度和映射文件。
- Qwen2 checkpoint、tokenizer 版本、扩展 token 列表和训练配置。

### 动手任务

1. 独立实现 Recall@K、NDCG@K 和碰撞率，并用手算样例测试。
2. 写一张完整消融表，包含假设、唯一变量、控制变量、指标和预期结论。
3. 用固定模板口述三个故障案例。
4. 运行 `python -m unittest discover -s tests -v` 和 `python -m compileall -q -f .`。

### 通关标准

- 能指出每个指标回答什么、不能回答什么。
- 能设计没有明显数据泄漏、候选不公平或变量混杂的实验。
- 能按固定顺序排障，并说明每一步如何缩小问题范围。

---

## 阶段 5：面试表达、手写题与完整模拟

### 阶段目标

- 把技术理解转化为稳定、可追问、有证据的项目表达。
- 对每个简历主张都能指向代码、命令、日志或实验记录。
- 在连续追问中保持术语、数据流和结果边界一致。

### 5.1 三版项目介绍

统一使用“背景 -> 问题 -> 方案 -> 个人工作 -> 验证 -> 局限与下一步”的结构，准备：

- 30 秒版：一句业务问题、一句三阶段方案、一句个人工作和边界。
- 2 分钟版：讲清三个阶段及关键设计选择。
- 5 分钟版：补充数据契约、指标、工程修复、消融和局限。

可作为简历描述的保守版本：

```text
实现 SASRec-RQ-VAE-Qwen2 三阶段生成式推荐原型，打通多模态商品表示、层次化
Semantic ID 和 next-item 生成训练的数据契约；重构硬编码路径、预训练权重加载、
目标标签掩码和碰撞推理，并补充环境自检与 GPU smoke test。
```

只有在你确实完成对应工作且可以举证时，才能使用这段话。不要直接写文档中的历史 loss、
2.2% 碰撞率或线上收益。

### 5.2 必须能独立手写

1. scaled dot-product attention 和 causal mask。
2. Recall@K、NDCG@K。
3. 最近邻量化和两层残差量化伪代码。
4. 将历史与 padding labels 设为 `-100` 的数据处理。
5. 按完整 code 分组并统计碰撞率。

通用算法题同步准备：哈希表、Top-K/堆、二分、滑动窗口、前缀和、LRU。推荐算法岗位
仍可能考通用数据结构；阶段学习期间每个学习单元搭配 1-2 道题即可，不以固定天数计数。

### 5.3 分段模拟与完整模拟

先分段练习：

1. 2 分钟项目介绍。
2. 5 分钟深入讲 SASRec。
3. 5 分钟深入讲 RQ-VAE。
4. 5 分钟深入讲 Qwen2 数据与推理。
5. 10 分钟回答评估、消融、上线与排障追问。

再完成一次 45-60 分钟完整模拟：项目概述、三阶段原理与代码、指标与系统设计、故障排查、
一道手写题。录音复盘时检查是否经常使用“可能、应该、大概”，并将模糊表达替换为有条件、
有证据的结论。

### 最终通关标准

- 不看资料画完整架构，标注主要 shape、loss、产物和指标。
- 连续回答至少 20 个项目问题，不混淆 ID、码本索引和 token。
- 能主动说明尚缺完整业务数据上的端到端对照实验与在线 Serving。
- 每个简历主张都有证据，未知内容明确说不知道并给出验证方法。

---

## 时间有限时如何裁剪

不要按天硬压缩，按能力裁剪：

1. **最小面试闭环**：阶段 0 全部；阶段 1 的数据构造、causal mask、BCE；阶段 2 的
   残差量化、loss、碰撞；阶段 3 的 label mask、tokenizer、合法解码；阶段 4 的指标和
   排障；阶段 5 的 2 分钟介绍与一次模拟。
2. **可以暂缓**：完整长时间训练、Sinkhorn 数学推导、DeepSpeed 内部实现、多机训练细节。
3. **不能省略**：手算或最小验证、离线指标、实验边界和口述复盘。面试中“看过但说不清”
   仍然等于没有掌握。

---

## 当前学习入口：阶段 2 RQ-VAE

阶段 0 和阶段 1 的基础学习已经完成，当前从阶段 2.1 继续。现阶段已经掌握：第一层量化
原始 latent，后续层依次量化上一层剩余的残差；各层码字之和构成最终量化向量。

已完成的手算例子：原始 latent 为 `[3,2]`，第一层码字为 `[2,1]`，第一层残差为
`[1,1]`；第二层码字为 `[1,0.5]`，最终量化向量为 `[3,1.5]`，剩余残差为 `[0,0.5]`。

接下来依次学习以下内容：

1. RQ-VAE 的 encoder、quantizer 和 decoder 分别承担什么职责。
2. 重建损失、codebook loss 和 commitment loss 各自约束谁靠近谁。
3. nearest-neighbor 选择为什么不可导，以及 straight-through estimator 如何传递梯度。
4. K-Means 初始化、死码、码本利用率、Sinkhorn 和碰撞率之间的关系。

本阶段不能只做到会算残差。通关时需要能结合 `rqvae/train/rqvae_model.py` 讲清完整前向
过程和损失，并结合 `rqvae/infer/rqvae_infer.py` 解释 Semantic ID 的生成、碰撞与回退策略。
