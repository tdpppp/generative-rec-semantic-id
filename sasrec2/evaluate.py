import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sasrec"))
from model import BaselineModel
from sasrec2.metrics import ranking_metrics
from sasrec2.temporal_dataset import TemporalSASRecDataset


def evaluate(args):
    cfg = SimpleNamespace(maxlen=args.maxlen, mm_emb_id=args.mm_emb_id, profile=args.profile, max_users=None,
                          cache_dir=None, rebuild_cache=False, device=args.device,
                          norm_first=False, hidden_units=args.hidden_units, num_heads=args.num_heads,
                          num_blocks=args.num_blocks, dropout_rate=args.dropout_rate, l2_emb=0.0)
    dataset = TemporalSASRecDataset(args.data_path, args.profile_path, cfg, args.split)
    model = BaselineModel(dataset.base.usernum, dataset.base.itemnum, dataset.base.feat_statistics, dataset.base.feature_types, cfg).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict); model.eval()
    item_ids = np.arange(1, dataset.base.itemnum + 1, dtype=np.int64)
    item_features = np.asarray([dataset.base.get_item_feature(int(item)) for item in item_ids], dtype=object)
    # Item representations are independent of the user, so compute them once.
    item_embeddings = []
    with torch.no_grad():
        for start in range(0, len(item_ids), args.item_batch_size):
            batch_ids = item_ids[start:start + args.item_batch_size]
            batch = torch.from_numpy(batch_ids).unsqueeze(0).to(args.device)
            feats = [item_features[start:start + len(batch_ids)]]
            emb = model.feat2emb(batch, feats, include_user=False).squeeze(0)
            item_embeddings.append(emb.detach().cpu())
    item_embeddings = torch.cat(item_embeddings, dim=0).to(args.device)
    print(f"precomputed item embeddings: {len(item_ids)} items, dim={item_embeddings.shape[-1]}", flush=True)
    ranks = []; predictions = []
    started_at = time.perf_counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            row = dataset.metadata(index)
            seq, _, _, mask, _, _, seq_feat, _, _ = dataset._sample(index, include_target=False)
            seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(args.device)
            user_vector = model.predict(seq_tensor, [seq_feat], torch.from_numpy(mask).unsqueeze(0)).squeeze(0)
            scores = torch.mv(item_embeddings, user_vector).cpu()
            top = torch.topk(scores, k=args.top_k).indices.numpy()
            predicted = [int(item_ids[position]) for position in top]
            target = int(row["target"])
            rank = next((i + 1 for i, item in enumerate(predicted) if item == target), None)
            if rank is not None: ranks.append(rank)
            predictions.append({"user_id": row["user_id"], "target_item_id": target, "rank": rank, "predicted_item_ids": predicted})
            if (index + 1) % 1000 == 0:
                elapsed = time.perf_counter() - started_at
                print(f"evaluated {index + 1}/{len(dataset)} users, {elapsed:.1f}s elapsed", flush=True)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "profile_path": str(Path(args.profile_path).resolve()),
        "split": args.split, "top_k": args.top_k, "metrics": ranking_metrics(ranks, len(dataset), prefix="item_"),
        "note": "Item-level SASRec ranking; Qwen metrics are Semantic-ID-level unless explicitly converted.",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.predictions:
        Path(args.predictions).write_text("\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--data_path", required=True); p.add_argument("--profile_path", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--split", choices=("validation", "test"), default="test"); p.add_argument("--output", required=True); p.add_argument("--predictions"); p.add_argument("--device", default="cuda:0"); p.add_argument("--mm_emb_id", nargs="+", default=["81", "82"]); p.add_argument("--profile", default="10k"); p.add_argument("--top_k", type=int, default=10); p.add_argument("--item_batch_size", type=int, default=4096); p.add_argument("--maxlen", type=int, default=100); p.add_argument("--hidden_units", type=int, default=32); p.add_argument("--num_heads", type=int, default=1); p.add_argument("--num_blocks", type=int, default=1); p.add_argument("--dropout_rate", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
