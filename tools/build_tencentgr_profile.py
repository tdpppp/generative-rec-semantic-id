#!/usr/bin/env python3
"""Build a reusable TencentGR-1M SASRec profile cache."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sasrec"))

from tencentgr_dataset import build_profile_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/TencentGR_1M")
    parser.add_argument("--profile", choices=["smoke", "10k", "100k", "full"], default="smoke")
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--mm_emb_id", nargs="+", choices=["81", "82"], default=["81", "82"])
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()
    path = build_profile_cache(
        data_path=args.data_path,
        profile=args.profile,
        max_users=args.max_users,
        mm_emb_ids=args.mm_emb_id,
        cache_dir=args.cache_dir,
        rebuild=args.rebuild_cache,
    )
    print(path)


if __name__ == "__main__":
    main()
