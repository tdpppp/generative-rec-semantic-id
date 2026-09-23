import argparse
import json
import os
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, HfArgumentParser, Trainer, TrainingArguments

try:
    from .arguments import DataTrainingArguments, ModelArguments
    from .custom_dataset import CustomTrainDataset
    from .utils import load_item2token_dict, load_model_and_tokenizer
except ImportError:
    from arguments import DataTrainingArguments, ModelArguments
    from custom_dataset import CustomTrainDataset
    from utils import load_item2token_dict, load_model_and_tokenizer


def parse_args_from_json(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        sections = json.load(stream)
    config = {}
    for section_name, values in sections.items():
        if not isinstance(values, dict):
            raise ValueError(f"Config section {section_name!r} must be an object")
        duplicate_keys = config.keys() & values.keys()
        if duplicate_keys:
            raise ValueError(f"Duplicate config keys: {sorted(duplicate_keys)}")
        config.update(values)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    return parser.parse_dict(config)


def resolve_cache_path(cache_root, value, name):
    if not value:
        raise ValueError(f"Missing required configuration value: {name}")
    path = Path(value).expanduser()
    return path if path.is_absolute() else cache_root / path


def main(model_args, data_args, training_args):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    cache_root = Path(os.environ.get("USER_CACHE_PATH", "./outputs")).expanduser()
    checkpoint_root = Path(os.environ.get("TRAIN_CKPT_PATH", "./outputs/gr/checkpoints")).expanduser()
    train_file = resolve_cache_path(cache_root, data_args.train_file, "train_file")
    mapping_dir = resolve_cache_path(cache_root, data_args.item2token_dict, "item2token_dict")
    if not train_file.exists():
        raise FileNotFoundError(f"Training data does not exist: {train_file}")

    output_name = Path(training_args.output_dir)
    output_dir = output_name if output_name.is_absolute() else checkpoint_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(output_dir)

    model, tokenizer = load_model_and_tokenizer(model_args)
    tokenizer.save_pretrained(output_dir)
    item2token_dict = load_item2token_dict(mapping_dir)
    train_dataset = CustomTrainDataset(
        json_path=str(train_file),
        item2token_dict=item2token_dict,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Train the generative recommender")
    cli_parser.add_argument("--config", required=True, help="Path to the JSON training config")
    cli_args = cli_parser.parse_args()
    main(*parse_args_from_json(cli_args.config))
