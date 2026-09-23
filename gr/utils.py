import os
from pathlib import Path

def _configured_path(value, fallback):
    path = Path(value).expanduser() if value else Path(fallback).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Required model path does not exist: {path}")
    return str(path)


def semantic_tokens(space_width):
    widths = [int(value.strip()) for value in space_width.split(",") if value.strip()]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError(f"Invalid se_id_space_width: {space_width!r}")
    if len(widths) > 26:
        raise ValueError("Semantic ID depth cannot exceed 26")

    tokens = ["<|hist_clk_start|>", "<|hist_clk_end|>"]
    for depth, width in enumerate(widths):
        prefix = chr(ord("a") + depth)
        tokens.extend(f"<{prefix}_{index}>" for index in range(width))
    return tokens


def load_model_and_tokenizer(model_args):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cache_root = Path(os.environ.get("USER_CACHE_PATH", "./outputs"))
    fallback_model = cache_root / "qwen_init2"
    configured_model = model_args.checkpoint_path if model_args.from_checkpoint else model_args.pretrained_path
    model_path = _configured_path(configured_model, fallback_model)
    tokenizer_path = _configured_path(model_args.tokenizer_path, model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.from_pretrained or model_args.from_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    added_count = tokenizer.add_special_tokens(
        {"additional_special_tokens": semantic_tokens(model_args.se_id_space_width)}
    )
    if added_count:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def load_item2token_dict(item2token_dict_path):
    mapping_dir = Path(item2token_dict_path)
    if not mapping_dir.is_dir():
        raise FileNotFoundError(f"Item-to-token directory does not exist: {mapping_dir}")

    item2token_dict = {}
    files = sorted(path for path in mapping_dir.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"No mapping files found in {mapping_dir}")
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item_id, token = line.split("\t", maxsplit=1)
                except ValueError as error:
                    raise ValueError(f"Invalid mapping at {path}:{line_number}") from error
                item2token_dict[item_id] = token
    print(f"length of item2token_dict is: {len(item2token_dict)}")
    return item2token_dict


def load_token2item_dict(token2item_dict_path):
    token2item_dict = {}
    with Path(token2item_dict_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                token, item_list = line.strip().split("\t", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Invalid token mapping at line {line_number}") from error
            token2item_dict[token] = item_list[1:-1].split(",")
    return token2item_dict
