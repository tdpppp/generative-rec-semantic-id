import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .custom_dataset import ProfileSequenceDataset
    from .utils import load_item2token_dict
except ImportError:
    from custom_dataset import ProfileSequenceDataset
    from utils import load_item2token_dict


SEMANTIC_ID_PATTERN = re.compile(r"^(<a_\d+>)(<b_\d+>)(<c_\d+>)$")


class SemanticIdConstraint:
    """Restrict three generated tokens to semantic IDs present in the mapping."""

    def __init__(self, tokenizer, semantic_ids, prompt_length):
        self.prompt_length = prompt_length
        self.first_tokens = set()
        self.second_tokens = defaultdict(set)
        self.third_tokens = defaultdict(set)
        self.code_by_ids = {}

        unknown_id = tokenizer.unk_token_id
        for code in semantic_ids:
            match = SEMANTIC_ID_PATTERN.fullmatch(code)
            if not match:
                raise ValueError(f"Invalid three-level Semantic ID: {code!r}")
            token_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in match.groups())
            if unknown_id is not None and unknown_id in token_ids:
                raise ValueError(f"Semantic ID token is missing from tokenizer: {code}")
            self.first_tokens.add(token_ids[0])
            self.second_tokens[token_ids[0]].add(token_ids[1])
            self.third_tokens[token_ids[:2]].add(token_ids[2])
            self.code_by_ids[token_ids] = code

        self.first_tokens = sorted(self.first_tokens)
        self.second_tokens = {key: sorted(values) for key, values in self.second_tokens.items()}
        self.third_tokens = {key: sorted(values) for key, values in self.third_tokens.items()}

    def allowed_tokens(self, batch_id, input_ids):
        generated = input_ids[self.prompt_length:].tolist()
        if not generated:
            return self.first_tokens
        if len(generated) == 1:
            return self.second_tokens.get(generated[0], [])
        if len(generated) == 2:
            return self.third_tokens.get(tuple(generated), [])
        return []

    def decode(self, token_ids):
        return self.code_by_ids.get(tuple(int(value) for value in token_ids))


def ranking_metrics(ranks, total):
    hits = len(ranks)
    return {
        "users": total,
        "hits": hits,
        "hr": hits / total if total else 0.0,
        "recall": hits / total if total else 0.0,
        "ndcg": sum(1.0 / math.log2(rank + 1) for rank in ranks) / total if total else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks) / total if total else 0.0,
    }


def resolve_num_beams(top_k, num_beams=None):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    resolved = top_k if num_beams is None else num_beams
    if resolved < top_k:
        raise ValueError(f"num_beams ({resolved}) must be greater than or equal to top_k ({top_k})")
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate constrained Semantic-ID generation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--profile_path", required=True)
    parser.add_argument("--mapping_dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, help="Beam-search width; defaults to top_k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", help="Optional JSON summary path")
    parser.add_argument("--predictions", help="Optional JSONL prediction path")
    args = parser.parse_args()
    try:
        args.num_beams = resolve_num_beams(args.top_k, args.num_beams)
    except ValueError as error:
        parser.error(str(error))
    return args


def evaluate(args):
    args.num_beams = resolve_num_beams(args.top_k, getattr(args, "num_beams", None))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device).eval()

    item2token = load_item2token_dict(args.mapping_dir)
    code_to_items = defaultdict(list)
    for item_id, code in item2token.items():
        code_to_items[code].append(int(item_id))
    data_args = SimpleNamespace(padding_side="left", max_seq_length=args.max_seq_length, token_depth=3)
    dataset = ProfileSequenceDataset(
        args.profile_path,
        item2token,
        tokenizer,
        data_args,
        split=args.split,
        max_samples=args.max_samples,
    )
    constraint = SemanticIdConstraint(tokenizer, code_to_items, prompt_length=0)

    ranks = []
    valid_format = 0
    valid_mapping = 0
    prediction_count = 0
    unique_predictions = set()
    ambiguous_targets = 0
    prediction_stream = None
    if args.predictions:
        prediction_path = Path(args.predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_stream = prediction_path.open("w", encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    evaluation_started = time.perf_counter()
    try:
        for start in range(0, len(dataset), args.batch_size):
            metadata = [dataset.example_metadata(index) for index in range(start, min(start + args.batch_size, len(dataset)))]
            prompts = []
            for row in metadata:
                history = "".join(item2token[str(item)] for item in row["history_item_ids"])
                prompts.append("<|hist_clk_start|>" + history + "<|hist_clk_end|>")
                ambiguous_targets += int(len(code_to_items[row["target_semantic_id"]]) > 1)
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=args.max_seq_length * 3 + 2,
                return_tensors="pt",
            ).to(device)
            constraint.prompt_length = encoded.input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=3,
                    min_new_tokens=3,
                    num_beams=args.num_beams,
                    num_return_sequences=args.top_k,
                    do_sample=False,
                    prefix_allowed_tokens_fn=constraint.allowed_tokens,
                    early_stopping=False,
                )
            generated = outputs[:, encoded.input_ids.shape[1]:].view(len(metadata), args.top_k, 3)
            for row, candidates in zip(metadata, generated):
                predicted_codes = []
                for candidate in candidates.tolist():
                    code = constraint.decode(candidate)
                    prediction_count += 1
                    valid_format += int(code is not None)
                    valid_mapping += int(code in code_to_items if code is not None else False)
                    if code is not None:
                        unique_predictions.add(code)
                    predicted_codes.append(code)
                target_code = row["target_semantic_id"]
                rank = next((index + 1 for index, code in enumerate(predicted_codes) if code == target_code), None)
                if rank is not None:
                    ranks.append(rank)
                if prediction_stream is not None:
                    prediction_stream.write(json.dumps({
                        "user_id": row["user_id"],
                        "target_item_id": row["target_item_id"],
                        "target_semantic_id": target_code,
                        "rank": rank,
                        "predicted_semantic_ids": predicted_codes,
                    }) + "\n")
    finally:
        if prediction_stream is not None:
            prediction_stream.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_cuda_memory = None
    evaluation_runtime = time.perf_counter() - evaluation_started

    result = {
        "model_path": str(Path(args.model_path).resolve()),
        "profile_path": str(Path(args.profile_path).resolve()),
        "mapping_dir": str(Path(args.mapping_dir).resolve()),
        "split": args.split,
        "max_seq_length": args.max_seq_length,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "evaluation_runtime_seconds": evaluation_runtime,
        "max_cuda_memory_allocated_bytes": peak_cuda_memory,
        "metrics": ranking_metrics(ranks, len(dataset)),
        "valid_format_rate": valid_format / prediction_count if prediction_count else 0.0,
        "valid_mapping_rate": valid_mapping / prediction_count if prediction_count else 0.0,
        "unique_recommended_semantic_ids": len(unique_predictions),
        "semantic_id_catalog_coverage": len(unique_predictions) / len(code_to_items) if code_to_items else 0.0,
        "ambiguous_target_rate": ambiguous_targets / len(dataset) if len(dataset) else 0.0,
        "note": "Ranking is evaluated at Semantic-ID level; collisions require a separate item-level tie-breaker.",
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate(parse_args())
