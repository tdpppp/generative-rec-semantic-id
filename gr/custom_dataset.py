import json
import warnings
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def encode_semantic_example(
    item_sequence,
    target_index,
    item2token_dict,
    tokenizer,
    max_seq_length,
    token_depth,
):
    target_item = str(item_sequence[target_index])
    target_token = item2token_dict.get(target_item)
    if target_token is None:
        return None
    history_tokens = [
        item2token_dict[str(item)]
        for item in item_sequence[:target_index]
        if str(item) in item2token_dict
    ][-max_seq_length:]
    if not history_tokens:
        return None

    history = "<|hist_clk_start|>" + "".join(history_tokens) + "<|hist_clk_end|>"
    full_input = history + target_token
    total_seq_length = max_seq_length * token_depth + token_depth + 2
    result = tokenizer(
        full_input,
        add_special_tokens=False,
        padding="max_length",
        max_length=total_seq_length,
        truncation=True,
    )
    history_length = len(tokenizer(history, add_special_tokens=False)["input_ids"])
    labels = list(result["input_ids"])
    for index in range(min(history_length, len(labels))):
        labels[index] = -100
    result["labels"] = [
        -100 if token_id == tokenizer.pad_token_id else label
        for token_id, label in zip(result["input_ids"], labels)
    ]
    return result


class CustomTrainDataset(IterableDataset):
    def __init__(self, json_path, item2token_dict, tokenizer, data_args):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.token_depth = data_args.token_depth
        self.tokenizer.padding_side = data_args.padding_side
        self.max_seq_length = data_args.max_seq_length
        self.item2token_dict = item2token_dict
        self.response_flag = data_args.response_flag
        self.data_format = getattr(data_args, "data_format", "auto")
        self.max_train_samples = getattr(data_args, "max_train_samples", None)
        self.total_seq_length = self.max_seq_length * self.token_depth + self.token_depth + 2

    def _stream_json_data(self):
        try:
            import ijson
        except ImportError:
            warnings.warn(
                "ijson is not installed; loading the complete training JSON into memory",
                RuntimeWarning,
            )
            with open(self.json_path, "r", encoding="utf-8") as stream:
                yield from json.load(stream).items()
            return

        with open(self.json_path, "rb") as stream:
            yield from ijson.kvitems(stream, "")

    def _stream_parquet_data(self):
        import pyarrow.parquet as pq

        path = Path(self.json_path)
        files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet sequence files found at {path}")
        emitted = 0
        for file in files:
            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(batch_size=2048, columns=["user_id", "seq"]):
                for row in batch.to_pylist():
                    sequence = [event["item_id"] for event in (row["seq"] or [])]
                    if len(sequence) < 2:
                        continue
                    yield str(row["user_id"]), sequence
                    emitted += 1
                    if self.max_train_samples is not None and emitted >= self.max_train_samples:
                        return

    def _stream_data(self):
        path = Path(self.json_path)
        data_format = self.data_format
        if data_format == "auto":
            data_format = "parquet" if path.is_dir() or path.suffix == ".parquet" else "json"
        if data_format == "parquet":
            yield from self._stream_parquet_data()
        elif data_format == "json":
            for index, record in enumerate(self._stream_json_data()):
                if self.max_train_samples is not None and index >= self.max_train_samples:
                    break
                yield record
        else:
            raise ValueError(f"Unsupported training data format: {data_format}")

    def _process_sequence(self, item_sequence):
        target_index = None
        target_item = None
        for index in range(len(item_sequence) - 1, -1, -1):
            candidate = str(item_sequence[index])
            if candidate in self.item2token_dict:
                target_index = index
                target_item = candidate
                break
        if target_index is None:
            return None
        return encode_semantic_example(
            item_sequence,
            target_index,
            self.item2token_dict,
            self.tokenizer,
            self.max_seq_length,
            self.token_depth,
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        shard_id = rank * worker_count + worker_id
        shard_count = world_size * worker_count

        for record_index, (_, item_sequence) in enumerate(self._stream_data()):
            if record_index % shard_count != shard_id or len(item_sequence) < 2:
                continue
            output = self._process_sequence(item_sequence)
            if output is not None:
                yield output


class ProfileSequenceDataset(Dataset):
    """Deterministic temporal splits backed by a TencentGR profile cache."""

    SPLITS = {"train", "validation", "test"}

    def __init__(self, profile_path, item2token_dict, tokenizer, data_args, split=None, max_samples=None):
        import pyarrow.parquet as pq

        self.profile_path = Path(profile_path)
        self.item2token_dict = item2token_dict
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.tokenizer.padding_side = data_args.padding_side
        self.max_seq_length = data_args.max_seq_length
        self.token_depth = data_args.token_depth
        self.split = split or getattr(data_args, "profile_split", "train")
        if self.split not in self.SPLITS:
            raise ValueError(f"Unsupported profile split: {self.split!r}")

        sequence_path = self.profile_path / "sequences.parquet"
        item_reid_path = self.profile_path / "item_reids.npy"
        if not sequence_path.is_file() or not item_reid_path.is_file():
            raise FileNotFoundError(
                f"Profile cache must contain sequences.parquet and item_reids.npy: {self.profile_path}"
            )
        item_reids = np.load(item_reid_path, mmap_mode="r")
        table = pq.read_table(sequence_path, columns=["reindexed_user_id", "seq"]).combine_chunks()

        self.user_ids = []
        self.sequences = []
        sample_counts = []
        for row in table.to_pylist():
            official_sequence = []
            for event in row["seq"] or []:
                local_id = int(event["item_id"])
                if local_id <= 0 or local_id >= len(item_reids):
                    raise ValueError(f"Invalid local item ID {local_id} in {sequence_path}")
                item_id = int(item_reids[local_id])
                if str(item_id) in item2token_dict:
                    official_sequence.append(item_id)
            count = max(0, len(official_sequence) - 3) if self.split == "train" else int(len(official_sequence) >= 3)
            if count:
                self.user_ids.append(int(row["reindexed_user_id"]))
                self.sequences.append(official_sequence)
                sample_counts.append(count)

        self.cumulative_counts = np.cumsum(sample_counts, dtype=np.int64)
        available_samples = int(self.cumulative_counts[-1]) if len(self.cumulative_counts) else 0
        self.sample_count = available_samples if max_samples is None else min(available_samples, max_samples)

    def __len__(self):
        return self.sample_count

    def _sample_coordinates(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sequence_index = bisect_right(self.cumulative_counts, index)
        previous_count = 0 if sequence_index == 0 else int(self.cumulative_counts[sequence_index - 1])
        offset = index - previous_count
        sequence = self.sequences[sequence_index]
        if self.split == "train":
            target_index = 1 + offset
        elif self.split == "validation":
            target_index = len(sequence) - 2
        else:
            target_index = len(sequence) - 1
        return sequence_index, target_index

    def example_metadata(self, index):
        sequence_index, target_index = self._sample_coordinates(index)
        sequence = self.sequences[sequence_index]
        target_item = str(sequence[target_index])
        return {
            "user_id": self.user_ids[sequence_index],
            "history_item_ids": sequence[:target_index][-self.max_seq_length:],
            "target_item_id": int(target_item),
            "target_semantic_id": self.item2token_dict[target_item],
        }

    def __getitem__(self, index):
        if self.tokenizer is None:
            raise RuntimeError("ProfileSequenceDataset requires a tokenizer to encode model inputs")
        sequence_index, target_index = self._sample_coordinates(index)
        result = encode_semantic_example(
            self.sequences[sequence_index],
            target_index,
            self.item2token_dict,
            self.tokenizer,
            self.max_seq_length,
            self.token_depth,
        )
        if result is None:
            raise RuntimeError(f"Profile sample {index} unexpectedly has no valid history or target")
        return result
