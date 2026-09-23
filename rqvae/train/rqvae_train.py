import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import CustomNpzFile
from rqvae_model import RQVAE
from trainer import Trainer


def default_cache_path(*parts):
    cache_root = Path(os.environ.get("USER_CACHE_PATH", "./outputs"))
    return str(cache_root.joinpath(*parts))


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RQ-VAE semantic-ID codebook")
    parser.add_argument("--data_path", default=default_cache_path("emb"), help="NPZ file or directory")
    parser.add_argument("--ckpt_dir", default=default_cache_path("rqvae", "checkpoints"))
    parser.add_argument("--log_dir", default=default_cache_path("rqvae", "tensorboard"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=20)
    parser.add_argument("--learner", default="AdamW")
    parser.add_argument("--lr_scheduler_type", choices=("constant", "linear"), default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss_type", choices=("mse", "l1"), default="mse")
    parser.add_argument("--kmeans_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[2048, 2048, 1024])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--quant_loss_weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--layers", type=int, nargs="+", default=[4096, 2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--save_limit", type=int, default=100)
    return parser.parse_args()


def discover_npz_files(data_path):
    path = Path(data_path).expanduser()
    if path.is_file() and path.suffix == ".npz":
        return [str(path)]
    if path.is_dir():
        files = sorted(str(item) for item in path.glob("*.npz"))
        if files:
            return files
    raise FileNotFoundError(f"No .npz files found at {path}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a CPU smoke run")
    if len(args.sk_epsilons) != len(args.num_emb_list):
        raise ValueError("--sk_epsilons and --num_emb_list must have the same number of values")

    seed_everything(args.seed)
    data = CustomNpzFile(discover_npz_files(args.data_path))
    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        shuffle=True,
        drop_last=False,
    )
    if len(data_loader) == 0:
        raise ValueError("The RQ-VAE dataset is empty")

    device = torch.device(args.device)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    ).to(device)

    trainer = Trainer(args, model, len(data_loader), device)
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f"best_loss={best_loss}")
    print(f"best_collision_rate={best_collision_rate}")


if __name__ == "__main__":
    main(parse_args())
