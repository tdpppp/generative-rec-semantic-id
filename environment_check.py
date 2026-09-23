#!/usr/bin/env python3
import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


REQUIRED_PACKAGES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("tensorboard", "tensorboard"),
    ("tqdm", "tqdm"),
    ("scikit-learn", "sklearn"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("ijson", "ijson"),
    ("pyarrow", "pyarrow"),
)


def status(ok, message):
    print(f"[{'OK' if ok else 'MISSING'}] {message}")
    return ok


def check_command(name):
    path = shutil.which(name)
    return status(path is not None, f"command {name}: {path or 'not found'}")


def check_packages():
    ok = True
    for distribution, module in REQUIRED_PACKAGES:
        available = importlib.util.find_spec(module) is not None
        version = importlib.metadata.version(distribution) if available else "not installed"
        ok = status(available, f"Python package {distribution}: {version}") and ok
    return ok


def check_cuda():
    if importlib.util.find_spec("torch") is None:
        return status(False, "CUDA through PyTorch: torch is not installed")
    import torch

    if not torch.cuda.is_available():
        return status(False, "CUDA through PyTorch: unavailable")
    return status(
        True,
        f"CUDA through PyTorch: {torch.cuda.get_device_name(0)} "
        f"(torch={torch.__version__}, cuda={torch.version.cuda})",
    )


def check_repository(root):
    expected = [
        root / "data" / "TencentGR_1M" / "seq",
        root / "data" / "TencentGR_1M" / "item_feat",
        root / "data" / "TencentGR_1M" / "user_feat",
        root / "data" / "TencentGR_1M" / "mm_emb" / "emb_81_32_parquet",
        root / "data" / "TencentGR_1M" / "mm_emb" / "emb_82_1024_parquet",
        root / "data" / "TencentGR_1M" / "indexer.pkl",
        root / "gr" / "gr_train.json",
    ]
    results = [status(path.exists(), f"repository input: {path.relative_to(root)}") for path in expected]
    return all(results)


def check_config(root):
    try:
        with (root / "gr" / "gr_train.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        widths = config["model_args"]["se_id_space_width"].split(",")
        depth = config["data_args"]["token_depth"]
        return status(len(widths) == depth, f"GR semantic-ID depth: widths={len(widths)}, token_depth={depth}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        return status(False, f"GR config: {error}")


def main():
    parser = argparse.ArgumentParser(description="Check the local LLM4Rec development environment")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when a required check fails")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    checks = [check_command("git"), check_command("nvidia-smi")]
    checks.append(check_packages())
    checks.append(check_cuda())
    checks.append(check_repository(root))
    checks.append(check_config(root))
    cache_root = Path(os.environ.get("USER_CACHE_PATH", root / "outputs"))
    checks.append(status((cache_root / "qwen_init2").is_dir(), f"Qwen2 weights: {cache_root / 'qwen_init2'}"))
    sequence_path = root / "data" / "TencentGR_1M" / "seq"
    checks.append(status(sequence_path.is_dir(), f"GR Parquet sequences: {sequence_path}"))
    mapping_dir = cache_root / "emb_infer" / "sinkhorn"
    mapping_ready = mapping_dir.is_dir() and any(mapping_dir.glob("*.txt"))
    checks.append(status(mapping_ready, f"Semantic-ID mapping: {mapping_dir}"))

    success = all(checks)
    print(f"Environment result: {'ready' if success else 'incomplete'}")
    if args.strict and not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
