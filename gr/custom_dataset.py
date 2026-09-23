import json
import warnings
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


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

        history_tokens = [
            self.item2token_dict[str(item)]
            for item in item_sequence[:target_index]
            if str(item) in self.item2token_dict
        ][-self.max_seq_length:]
        if not history_tokens:
            return None

        history = "<|hist_clk_start|>" + "".join(history_tokens) + "<|hist_clk_end|>"
        full_input = history + self.item2token_dict[target_item]
        result = self.tokenizer(
            full_input,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.total_seq_length,
            truncation=True,
        )
        history_length = len(self.tokenizer(history, add_special_tokens=False)["input_ids"])
        labels = list(result["input_ids"])
        for index in range(min(history_length, len(labels))):
            labels[index] = -100
        labels = [
            -100 if token_id == self.tokenizer.pad_token_id else label
            for token_id, label in zip(result["input_ids"], labels)
        ]
        result["labels"] = labels
        return result

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
