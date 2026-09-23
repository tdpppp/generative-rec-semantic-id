"""
SASRec 多模态序列表征学习 —— 训练入口

本文件是整个三阶段生成式推荐链路的第一阶段（Stage 1）：
  1. 从 TencentGR-1M Parquet profile 加载用户行为序列与物品特征；
  2. 训练一个融合多模态信息的 Transformer 序列模型（BaselineModel）；
  3. 训练完成后，调用 model.save_item_emb 把每个物品编码成 32 维稠密向量，
     保存为 embeddings.npz，供第二阶段 RQ-VAE 构建语义 ID 使用。

运行依赖以下环境变量（均在 __main__ 块中读取）：
  TRAIN_DATA_PATH      TencentGR-1M 数据目录
  TRAIN_LOG_PATH       训练日志目录（写入 train.log）
  TRAIN_TF_EVENTS_PATH TensorBoard 事件目录
  TRAIN_CKPT_PATH      模型权重保存目录（model.pt）
  USER_CACHE_PATH      结果缓存根目录（物品向量导出到其下的 emb/embeddings.npz）
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import BaselineModel
from tencentgr_dataset import TencentGRDataset


def get_args():
    """解析命令行参数。

    注意：数据路径/日志路径等 IO 相关配置不在这里，而是通过环境变量传入，
    这里的参数主要控制模型结构与训练超参。
    """
    parser = argparse.ArgumentParser()

    # ===== 训练超参 =====
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)  # 序列最大长度

    # ===== Baseline 模型结构 =====
    parser.add_argument('--hidden_units', default=32, type=int)   # 表征维度（最终物品向量维度）
    parser.add_argument('--num_blocks', default=1, type=int)      # Transformer 层数
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--max_steps', default=None, type=int, help='stop after this many optimizer steps')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)      # embedding 的 L2 正则系数（默认关闭）
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')  # 仅推理不训练（会直接跳过训练循环）
    parser.add_argument('--state_dict_path', default=None, type=str)  # 从已有权重继续训练
    parser.add_argument('--norm_first', action='store_true')      # Transformer 是否 Pre-LN

    # IO paths: command line values take precedence over environment variables.
    parser.add_argument('--data_path', default=os.environ.get('TRAIN_DATA_PATH'))
    parser.add_argument('--profile', choices=['smoke', '10k', '100k', 'full'], default='smoke')
    parser.add_argument('--max_users', default=None, type=int, help='override the profile user limit')
    parser.add_argument('--cache_dir', default=None, help='optional directory for generated profile caches')
    parser.add_argument('--rebuild_cache', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--log_dir', default=os.environ.get('TRAIN_LOG_PATH', './outputs/sasrec/logs'))
    parser.add_argument(
        '--tensorboard_dir',
        default=os.environ.get('TRAIN_TF_EVENTS_PATH', './outputs/sasrec/tensorboard'),
    )
    parser.add_argument('--ckpt_dir', default=os.environ.get('TRAIN_CKPT_PATH', './outputs/sasrec/checkpoints'))
    parser.add_argument(
        '--emb_output_dir',
        default=os.path.join(os.environ.get('USER_CACHE_PATH', './outputs'), 'emb'),
    )

    # ===== 多模态特征 ID =====
    # 有 6 种多模态特征（81-86），默认只用 81/82，可按需增减（需数据目录里有对应特征）
    parser.add_argument('--mm_emb_id', nargs='+', default=['81', '82'], type=str, choices=['81', '82'])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if not args.data_path:
        raise ValueError('Missing data path: pass --data_path or set TRAIN_DATA_PATH')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; use --device cpu for a CPU smoke run')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- 1. 准备日志与 TensorBoard 输出目录 ----
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(args.log_dir, 'train.log'), 'w', encoding='utf-8')
    writer = SummaryWriter(args.tensorboard_dir)

    # ---- 2. 确定数据路径与结果缓存路径 ----
    # data_path：训练数据目录；emb_save_dir：物品向量最终导出目录的根（见末尾 save_item_emb）
    data_path = args.data_path
    emb_save_dir = Path(args.emb_output_dir)

    # ---- 3. 构建数据集与 DataLoader ----
    dataset = TencentGRDataset(data_path, args)
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1], generator=split_generator
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )
    # 从数据集取用户数/物品数/特征统计，供模型声明 embedding 表维度
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # ---- 4. 构建模型并初始化 ----
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 参数用 Xavier 初始化；对无 bias 的层（如 Embedding）会捕获异常跳过
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # 关键：把所有 padding_idx=0 的 embedding 第 0 行手动置 0，
    # 保证 padding 位置的向量恒为 0，不参与梯度与表示
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    # ---- 5. （可选）从已有 checkpoint 恢复训练 ----
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            # 从文件名解析起始 epoch，形如 "...epoch=N.pth"
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # ---- 6. 定义损失与优化器 ----
    # 负采样 + BCE：正样本标签为 1，负样本标签为 0
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")

    # ---- 7. 训练 + 验证主循环 ----
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break  # 仅推理模式：跳过训练

        # ---------- 训练阶段 ----------
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 解包 dataset.collate_fn 返回的 9 个字段：
            #   seq/pos/neg 分别是当前序列、正样本（下一真实 item）、负样本（随机采样）的 ID
            #   token_type/next_token_type 标识 token 是 user(2) 还是 item(1)
            #   next_action_type 标识下一动作是曝光(0)还是点击(1)
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)

            # 前向：pos_logits / neg_logits 是「序列表征」与「正/负样本表征」的内积打分
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )

            # 标签：正样本全 1，负样本全 0
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()

            # 只对「下一个 token 是 item(type=1)」的位置计算 loss，
            # 屏蔽预测 user token 的位置（序列推荐只关心预测物品）
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # 写训练日志（每步一行 JSON）与 TensorBoard
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

            # 可选：item embedding 的 L2 正则（默认 l2_emb=0.0 不生效）
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        # ---------- 验证阶段 ----------
        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                indices = np.where(next_token_type == 1)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                valid_loss_sum += loss.item()
        valid_loss_sum /= max(len(valid_loader), 1)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        # ---------- 保存模型权重 ----------
        # 每个 epoch 保存一次，目录名带步数与验证 loss，便于回溯
        save_dir = Path(args.ckpt_dir, f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

        # ---------- 导出物品向量（供 RQ-VAE 使用） ----------
        if epoch % 1 == 0:  # 每个 epoch 都导出，可按需调整频率
            print("Saving item embeddings...")

            # 准备 item ID：从 1 开始（0 是 padding）
            item_ids = list(range(1, itemnum + 1))
            reindexed_ids = np.asarray(dataset.item_reids[1:], dtype=np.int64)
            original_ids = np.asarray(dataset.item_original_ids[1:], dtype=np.int64)

            # 物品向量导出目录：USER_CACHE_PATH/emb
            emb_save_dir.mkdir(parents=True, exist_ok=True)

            # 调用 save_item_emb：对每个物品做 feat2emb（include_user=False），
            # 编码成 32 维向量并保存为 embeddings.npz（键：ids / embs）
            model.save_item_emb(
                item_ids=item_ids,
                output_ids=reindexed_ids,
                original_ids=original_ids,
                feat_dict=dataset.item_feature_view,
                save_path=emb_save_dir,
                batch_size=args.batch_size
            )
            print(f"Item embeddings saved to {emb_save_dir}")

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print("Done")
    writer.close()
    log_file.close()
