import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

try:
    from .custom_dataset import ProfileSequenceDataset
    from .evaluate_gr import SEMANTIC_ID_PATTERN, ranking_metrics
    from .utils import load_item2token_dict
except ImportError:
    from custom_dataset import ProfileSequenceDataset
    from evaluate_gr import SEMANTIC_ID_PATTERN, ranking_metrics
    from utils import load_item2token_dict


def build_most_popular_candidates(sequences, item2token, top_k):
    """Rank codes using only events before the validation and test holdouts."""
    catalog = sorted(set(item2token.values()))
    if top_k <= 0 or top_k > len(catalog):
        raise ValueError(f"top_k must be between 1 and the catalog size ({len(catalog)})")

    counts = Counter()
    for sequence in sequences:
        for item_id in sequence[:-2]:
            code = item2token.get(str(item_id))
            if code is not None:
                counts[code] += 1

    ranked = sorted(catalog, key=lambda code: (-counts[code], code))
    return ranked[:top_k]


def sample_random_candidates(catalog, top_k, rng):
    if top_k <= 0 or top_k > len(catalog):
        raise ValueError(f"top_k must be between 1 and the catalog size ({len(catalog)})")
    return rng.sample(catalog, top_k)


def evaluate_candidate_ranking(
    name,
    dataset,
    candidate_provider,
    code_to_items,
    top_k,
    profile_path,
    mapping_dir,
    predictions_path,
):
    ranks = []
    valid_format = 0
    valid_mapping = 0
    prediction_count = 0
    unique_predictions = set()
    ambiguous_targets = 0

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as stream:
        for index in range(len(dataset)):
            row = dataset.example_metadata(index)
            candidates = list(candidate_provider(index, row))
            if len(candidates) != top_k:
                raise ValueError(f"{name} returned {len(candidates)} candidates; expected {top_k}")
            if len(set(candidates)) != top_k:
                raise ValueError(f"{name} returned duplicate candidates for user {row['user_id']}")

            for code in candidates:
                prediction_count += 1
                valid_format += int(SEMANTIC_ID_PATTERN.fullmatch(code) is not None)
                valid_mapping += int(code in code_to_items)
                unique_predictions.add(code)

            target_code = row["target_semantic_id"]
            ambiguous_targets += int(len(code_to_items[target_code]) > 1)
            rank = next((position + 1 for position, code in enumerate(candidates) if code == target_code), None)
            if rank is not None:
                ranks.append(rank)

            stream.write(json.dumps({
                "user_id": row["user_id"],
                "target_item_id": row["target_item_id"],
                "target_semantic_id": target_code,
                "rank": rank,
                "predicted_semantic_ids": candidates,
            }) + "\n")

    return {
        "baseline": name,
        "profile_path": str(Path(profile_path).resolve()),
        "mapping_dir": str(Path(mapping_dir).resolve()),
        "split": dataset.split,
        "top_k": top_k,
        "metrics": ranking_metrics(ranks, len(dataset)),
        "valid_format_rate": valid_format / prediction_count if prediction_count else 0.0,
        "valid_mapping_rate": valid_mapping / prediction_count if prediction_count else 0.0,
        "unique_recommended_semantic_ids": len(unique_predictions),
        "semantic_id_catalog_coverage": len(unique_predictions) / len(code_to_items) if code_to_items else 0.0,
        "ambiguous_target_rate": ambiguous_targets / len(dataset) if len(dataset) else 0.0,
        "note": "Ranking is evaluated at Semantic-ID level; popularity excludes validation and test holdouts.",
    }


def write_metrics(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate non-neural baselines for generative recommendation")
    parser.add_argument("--profile_path", required=True)
    parser.add_argument("--mapping_dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument(
        "--output_dir",
        default="outputs/gr/baselines/10k_balanced/semantic_id_top10",
    )
    return parser.parse_args()


def evaluate(args):
    item2token = load_item2token_dict(args.mapping_dir)
    code_to_items = defaultdict(list)
    for item_id, code in item2token.items():
        code_to_items[code].append(int(item_id))

    data_args = SimpleNamespace(padding_side="right", max_seq_length=args.max_seq_length, token_depth=3)
    dataset = ProfileSequenceDataset(
        args.profile_path,
        item2token,
        tokenizer=None,
        data_args=data_args,
        split=args.split,
        max_samples=args.max_samples,
    )
    catalog = sorted(code_to_items)
    popular_candidates = build_most_popular_candidates(dataset.sequences, item2token, args.top_k)
    output_dir = Path(args.output_dir)

    most_popular = evaluate_candidate_ranking(
        "most_popular",
        dataset,
        lambda _index, _row: popular_candidates,
        code_to_items,
        args.top_k,
        args.profile_path,
        args.mapping_dir,
        output_dir / "most_popular_predictions.jsonl",
    )
    write_metrics(output_dir / "most_popular_metrics.json", most_popular)

    rng = random.Random(args.seed)
    random_result = evaluate_candidate_ranking(
        "random",
        dataset,
        lambda _index, _row: sample_random_candidates(catalog, args.top_k, rng),
        code_to_items,
        args.top_k,
        args.profile_path,
        args.mapping_dir,
        output_dir / "random_predictions.jsonl",
    )
    random_result["seed"] = args.seed
    write_metrics(output_dir / "random_metrics.json", random_result)

    comparison = {
        "most_popular": most_popular,
        "random": random_result,
    }
    write_metrics(output_dir / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return comparison


if __name__ == "__main__":
    evaluate(parse_args())
