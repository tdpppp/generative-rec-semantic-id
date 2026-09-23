import argparse
import collections
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CustomNpzFile
from rqvae_model import RQVAE


def default_cache_path(*parts):
    return str(Path(os.environ.get("USER_CACHE_PATH", "./outputs")).joinpath(*parts))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate semantic IDs with a trained RQ-VAE")
    parser.add_argument("--data_path", default=default_cache_path("emb"), help="NPZ file or directory")
    parser.add_argument("--checkpoint", required=True, help="RQ-VAE checkpoint produced by training")
    parser.add_argument("--output_dir", default=default_cache_path("emb_infer", "sinkhorn"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=81920)
    parser.add_argument("--collision_batch_size", type=int, default=8192)
    parser.add_argument("--max_collision_rounds", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.003)
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


def code_string(index, prefixes):
    if len(index) > len(prefixes):
        raise ValueError(f"Semantic ID depth {len(index)} exceeds supported depth {len(prefixes)}")
    return "".join(prefixes[level].format(int(value)) for level, value in enumerate(index))


def collision_groups(item_to_code):
    code_to_items = collections.defaultdict(list)
    for item_id, code in item_to_code.items():
        code_to_items[code].append(item_id)
    return [items for items in code_to_items.values() if len(items) > 1]


def build_model(checkpoint_path, input_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    model = RQVAE(
        in_dim=input_dim,
        num_emb_list=model_args.num_emb_list,
        e_dim=model_args.e_dim,
        layers=model_args.layers,
        dropout_prob=model_args.dropout_prob,
        bn=model_args.bn,
        loss_type=model_args.loss_type,
        quant_loss_weight=model_args.quant_loss_weight,
        beta=getattr(model_args, "beta", 0.25),
        kmeans_init=model_args.kmeans_init,
        kmeans_iters=model_args.kmeans_iters,
        sk_epsilons=model_args.sk_epsilons,
        sk_iters=model_args.sk_iters,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def infer(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    device = torch.device(args.device)
    data = CustomNpzFile(discover_npz_files(args.data_path))
    loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    model = build_model(args.checkpoint, data.dim, device)
    prefixes = [f"<{chr(ord('a') + level)}_{{}}>" for level in range(26)]

    item_to_code = {}
    item_to_dataset_index = {}
    dataset_index = 0
    for item_ids, embeddings in tqdm(loader, desc="Initial quantization"):
        indices = model.get_indices(embeddings.to(device), use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().tolist()
        ids = item_ids.view(-1).cpu().tolist()
        if len(ids) != len(indices):
            raise RuntimeError("RQ-VAE returned a different number of codes than input items")
        for item_id, index in zip(ids, indices):
            item_id = int(item_id)
            item_to_code[item_id] = code_string(index, prefixes)
            item_to_dataset_index[item_id] = dataset_index
            dataset_index += 1

    for quantizer in model.rq.vq_layers[:-1]:
        quantizer.sk_epsilon = 0.0
    model.rq.vq_layers[-1].sk_epsilon = args.sinkhorn_epsilon

    for round_index in range(args.max_collision_rounds):
        groups = collision_groups(item_to_code)
        if not groups:
            print(f"All semantic IDs are unique after {round_index} collision rounds")
            break
        collision_items = [item_id for group in groups for item_id in group]
        print(f"collision_round={round_index + 1} groups={len(groups)} items={len(collision_items)}")

        for start in tqdm(
            range(0, len(collision_items), args.collision_batch_size),
            desc="Resolving collisions",
        ):
            expected_ids = collision_items[start:start + args.collision_batch_size]
            dataset_indices = [item_to_dataset_index[item_id] for item_id in expected_ids]
            actual_ids, embeddings = data[dataset_indices]
            actual_ids = [int(item_id) for item_id in actual_ids.view(-1).tolist()]
            if actual_ids != expected_ids:
                raise RuntimeError("Dataset item order changed while resolving semantic-ID collisions")

            indices = model.get_indices(embeddings.to(device), use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().tolist()
            for item_id, index in zip(actual_ids, indices):
                item_to_code[item_id] = code_string(index, prefixes)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    unique_count = len(set(item_to_code.values()))
    item_count = len(item_to_code)
    collision_rate = (item_count - unique_count) / item_count if item_count else 0.0
    print(f"items={item_count} unique_codes={unique_count} collision_rate={collision_rate:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "worker_0_output.txt"
    with output_file.open("w", encoding="utf-8") as stream:
        for item_id, code in sorted(item_to_code.items()):
            stream.write(f"{item_id}\t{code}\n")
    print(f"Semantic-ID mapping written to {output_file}")


if __name__ == "__main__":
    infer(parse_args())
