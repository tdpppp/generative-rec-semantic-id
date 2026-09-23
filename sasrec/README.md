本目录直接读取官方 TencentGR-1M Parquet 数据。`tencentgr_dataset.py` 负责构建
`smoke`、`10k`、`100k` 或 `full` profile，使用局部连续 ID 控制模型规模，同时保存
官方 re-ID 和原始商品 ID。`main.py` 训练 SASRec，`model.py` 的 `save_item_emb` 为后续
RQ-VAE 导出商品 embedding。运行参数与完整命令见根目录 `README.md`。
