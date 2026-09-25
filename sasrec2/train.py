import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sasrec"))
from model import BaselineModel
from sasrec2.temporal_dataset import TemporalSASRecDataset


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = SimpleNamespace(maxlen=args.maxlen, mm_emb_id=args.mm_emb_id, profile=args.profile, max_users=None,
                          cache_dir=None, rebuild_cache=False, device=args.device,
                          norm_first=False, hidden_units=args.hidden_units, num_heads=args.num_heads,
                          num_blocks=args.num_blocks, dropout_rate=args.dropout_rate, l2_emb=0.0)
    train = TemporalSASRecDataset(args.data_path, args.profile_path, cfg, "train")
    valid = TemporalSASRecDataset(args.data_path, args.profile_path, cfg, "validation")
    model = BaselineModel(train.base.usernum, train.base.itemnum, train.base.feat_statistics, train.base.feature_types, cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    criterion = torch.nn.BCEWithLogitsLoss()
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "collate_fn": train.collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and str(args.device).startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(train, **loader_kwargs)
    out = Path(args.output_dir); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    best = float("inf")
    global_step = 0
    stop_training = False
    for epoch in range(1, args.num_epochs + 1):
        model.train(); total = 0.0
        epoch_start = time.perf_counter()
        progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
        for batch in progress:
            seq, pos, neg, mask, next_mask, action, seq_feat, pos_feat, neg_feat = batch
            seq, pos, neg = seq.to(args.device), pos.to(args.device), neg.to(args.device)
            p, n = model(seq, pos, neg, mask, next_mask, action, seq_feat, pos_feat, neg_feat)
            active = next_mask.to(args.device) == 1
            loss = criterion(p[active], torch.ones_like(p[active])) + criterion(n[active], torch.zeros_like(n[active]))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item())
            global_step += 1
            if global_step == 1 or global_step % args.log_steps == 0:
                elapsed = max(time.perf_counter() - epoch_start, 1e-6)
                samples_per_second = global_step * args.batch_size / elapsed
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(loss.item()),
                    "samples_per_second": samples_per_second,
                }
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    record["gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                print(json.dumps(record), flush=True)
            if args.max_steps is not None and global_step >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            print(json.dumps({"status": "stopped_at_max_steps", "step": global_step}), flush=True)
            break
        model.eval(); valid_loss = 0.0
        with torch.no_grad():
            valid_kwargs = {
                "batch_size": args.batch_size,
                "collate_fn": valid.collate_fn,
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory and str(args.device).startswith("cuda"),
            }
            if args.num_workers > 0:
                valid_kwargs["persistent_workers"] = True
                valid_kwargs["prefetch_factor"] = args.prefetch_factor
            for batch in DataLoader(valid, **valid_kwargs):
                seq, pos, neg, mask, next_mask, action, seq_feat, pos_feat, neg_feat = batch
                p, n = model(seq.to(args.device), pos.to(args.device), neg.to(args.device), mask, next_mask, action, seq_feat, pos_feat, neg_feat)
                active = next_mask.to(args.device) == 1
                valid_loss += float((criterion(p[active], torch.ones_like(p[active])) + criterion(n[active], torch.zeros_like(n[active]))).item())
        valid_loss /= max(len(valid), 1)
        checkpoint = out / "checkpoints" / f"epoch-{epoch}.valid_loss={valid_loss:.6f}.pt"
        torch.save(model.state_dict(), checkpoint)
        if valid_loss < best:
            best = valid_loss
            torch.save(model.state_dict(), out / "checkpoints" / "best_model.pt")
        print(json.dumps({"epoch": epoch, "train_loss": total / max(len(loader), 1), "valid_loss": valid_loss}))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True); p.add_argument("--profile_path", required=True); p.add_argument("--output_dir", required=True); p.add_argument("--profile", default="10k")
    p.add_argument("--mm_emb_id", nargs="+", default=["81", "82"]); p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256); p.add_argument("--num_epochs", type=int, default=3); p.add_argument("--max_steps", type=int, default=None); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--maxlen", type=int, default=100); p.add_argument("--hidden_units", type=int, default=32); p.add_argument("--num_heads", type=int, default=1); p.add_argument("--num_blocks", type=int, default=1); p.add_argument("--dropout_rate", type=float, default=0.2); p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num_workers", type=int, default=4); p.add_argument("--prefetch_factor", type=int, default=2); p.add_argument("--log_steps", type=int, default=20); p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
