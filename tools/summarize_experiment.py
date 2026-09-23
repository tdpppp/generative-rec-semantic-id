#!/usr/bin/env python3
"""Summarize SASRec and RQ-VAE artifacts from one experiment directory."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


VALID_LOSS_PATTERN = re.compile(r"valid_loss=([0-9.]+)")
SEMANTIC_TOKEN_PATTERN = re.compile(r"<([a-z])_(\d+)>")


def summarize_sasrec(run_dir):
    result = {}
    log_path = run_dir / "sasrec" / "logs" / "train.log"
    if log_path.is_file():
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        losses = [float(row["loss"]) for row in rows]
        if losses:
            result["train"] = {
                "steps": len(losses),
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "minimum_loss": min(losses),
                "mean_loss": sum(losses) / len(losses),
            }
    checkpoint_root = run_dir / "sasrec" / "checkpoints"
    validation = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.iterdir():
            match = VALID_LOSS_PATTERN.search(path.name)
            if match:
                validation.append({"path": str(path), "loss": float(match.group(1))})
    if validation:
        result["validation"] = sorted(validation, key=lambda item: item["loss"])
    embedding_path = run_dir / "emb" / "embeddings.npz"
    if embedding_path.is_file():
        data = np.load(embedding_path)
        embeddings = data["embs"]
        ids = data["ids"]
        result["embeddings"] = {
            "path": str(embedding_path),
            "items": int(len(ids)),
            "dimension": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "unique_ids": int(len(np.unique(ids))),
            "all_finite": bool(np.isfinite(embeddings).all()),
            "keys": list(data.files),
        }
    return result


def summarize_tensorboard(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"error": "tensorboard is not installed"}
    event_files = sorted((run_dir / "rqvae" / "tensorboard").glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    result = {"event_file": str(event_files[-1])}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)
        if not values:
            continue
        result[tag] = {
            "count": len(values),
            "first": float(values[0].value),
            "last": float(values[-1].value),
            "minimum": float(min(value.value for value in values)),
        }
    return result


def summarize_semantic_ids(run_dir):
    candidates = sorted((run_dir / "emb_infer").glob("*/worker_0_output.txt"))
    if not candidates:
        return {}
    result = {}
    for path in candidates:
        item_ids = []
        codes = []
        level_values = {}
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    item_id, code = line.rstrip().split("\t", 1)
                except ValueError as error:
                    raise ValueError(f"Invalid mapping at {path}:{line_number}") from error
                item_ids.append(int(item_id))
                codes.append(code)
                for prefix, value in SEMANTIC_TOKEN_PATTERN.findall(code):
                    level_values.setdefault(prefix, set()).add(int(value))
        unique_codes = len(set(codes))
        count = len(codes)
        result[path.parent.name] = {
            "path": str(path),
            "items": count,
            "unique_item_ids": len(set(item_ids)),
            "unique_semantic_ids": unique_codes,
            "collision_count": count - unique_codes,
            "collision_rate": (count - unique_codes) / count if count else None,
            "codebook_usage": {prefix: len(values) for prefix, values in sorted(level_values.items())},
        }
    return result


def summarize(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "sasrec": summarize_sasrec(run_dir),
        "rqvae_tensorboard": summarize_tensorboard(run_dir),
        "semantic_ids": summarize_semantic_ids(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="outputs/10k")
    parser.add_argument("--output", help="optional JSON file to write")
    args = parser.parse_args()
    report = summarize(args.run_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
