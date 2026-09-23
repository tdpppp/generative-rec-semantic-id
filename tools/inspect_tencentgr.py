#!/usr/bin/env python3
"""Inspect the official TencentGR-1M Parquet dataset without loading it all."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


DATASETS = {
    "seq": "seq",
    "item_feat": "item_feat",
    "user_feat": "user_feat",
    "candidate": "candidate",
    "emb_81": "mm_emb/emb_81_32_parquet",
    "emb_82": "mm_emb/emb_82_1024_parquet",
}


def inspect_directory(path):
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {path}")
    metadata = [pq.ParquetFile(file).metadata for file in files]
    first = pq.ParquetFile(files[0])
    sample = first.read_row_group(0).slice(0, 1).to_pylist()[0]
    return {
        "path": str(path),
        "files": len(files),
        "rows": sum(item.num_rows for item in metadata),
        "schema": str(first.schema_arrow),
        "sample_summary": {
            key: f"list[{len(value)}]" if isinstance(value, list) else value
            for key, value in sample.items()
            if key != "emb"
        },
        "embedding_length": len(sample["emb"]) if sample.get("emb") is not None else None,
    }


def inspect_dataset(root):
    root = Path(root).expanduser().resolve()
    required = [root / relative for relative in DATASETS.values()] + [root / "indexer.pkl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"TencentGR-1M is incomplete; missing: {missing}")
    return {name: inspect_directory(root / relative) for name, relative in DATASETS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/TencentGR_1M")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = inspect_dataset(args.data_path)
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2, default=str))
        return
    for name, details in report.items():
        print(f"[{name}] files={details['files']} rows={details['rows']}")
        print(details["schema"])
        print(f"sample={details['sample_summary']}")
        if details["embedding_length"] is not None:
            print(f"embedding_length={details['embedding_length']}")


if __name__ == "__main__":
    main()
