"""TencentGR-1M Parquet profile builder and SASRec dataset."""

import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


PROFILE_USERS = {"smoke": 256, "10k": 10_000, "100k": 100_000, "full": None}
ITEM_SPARSE = ["100", "117", "118", "101", "102", "119", "120", "114", "112", "121", "115", "122", "116"]
USER_SPARSE = ["103", "104", "105", "109"]
USER_ARRAY = ["106", "107", "108", "110"]
EMBEDDING_DIMS = {"81": 32, "82": 1024}


def _parquet_files(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {path}")
    return files


def _iter_rows(path, columns, batch_size=2048):
    for file in _parquet_files(path):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def _write_rows(writer, rows, schema):
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def _cache_name(profile, max_users, mm_emb_ids):
    limit = "all" if max_users is None else str(max_users)
    return f"{profile}-u{limit}-mm{'-'.join(mm_emb_ids)}"


def resolve_profile_cache(data_path, profile, max_users, mm_emb_ids, cache_dir=None):
    if profile not in PROFILE_USERS:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILE_USERS)}")
    effective_max_users = PROFILE_USERS[profile] if max_users is None else max_users
    root = Path(cache_dir) if cache_dir else Path(data_path) / "cache" / "profiles"
    return root / _cache_name(profile, effective_max_users, mm_emb_ids), effective_max_users


def _source_is_complete(data_path, mm_emb_ids):
    required = ["seq", "item_feat", "user_feat", "indexer.pkl"]
    required.extend(f"mm_emb/emb_{feature_id}_{EMBEDDING_DIMS[feature_id]}_parquet" for feature_id in mm_emb_ids)
    missing = [relative for relative in required if not (Path(data_path) / relative).exists()]
    if missing:
        raise FileNotFoundError(f"TencentGR-1M input is incomplete; missing {missing}")


def _load_index_metadata(data_path, selected_item_reids):
    with (Path(data_path) / "indexer.pkl").open("rb") as stream:
        indexer = pickle.load(stream)
    feature_cardinalities = {str(key): len(value) for key, value in indexer["f"].items()}
    selected = set(selected_item_reids)
    reid_to_original = {
        int(reid): int(original)
        for original, reid in indexer["i"].items()
        if int(reid) in selected
    }
    if len(reid_to_original) != len(selected):
        missing = sorted(selected - reid_to_original.keys())[:10]
        raise ValueError(f"indexer.pkl has no original item ID for re-IDs {missing}")
    return feature_cardinalities, reid_to_original


def build_profile_cache(data_path, profile="smoke", max_users=None, mm_emb_ids=("81", "82"), cache_dir=None, rebuild=False):
    """Build a compact, deterministic training profile from TencentGR-1M."""
    data_path = Path(data_path).expanduser().resolve()
    mm_emb_ids = tuple(str(item) for item in mm_emb_ids)
    unsupported = set(mm_emb_ids) - EMBEDDING_DIMS.keys()
    if unsupported:
        raise ValueError(f"Only downloaded embedding features 81/82 are supported, got {sorted(unsupported)}")
    _source_is_complete(data_path, mm_emb_ids)
    cache_path, effective_max_users = resolve_profile_cache(
        data_path, profile, max_users, mm_emb_ids, cache_dir
    )
    metadata_path = cache_path / "metadata.json"
    if metadata_path.is_file() and not rebuild:
        return cache_path
    if cache_path.exists() and not rebuild:
        raise RuntimeError(f"Incomplete profile cache exists at {cache_path}; pass --rebuild_cache")
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{cache_path.name}-", dir=cache_path.parent))

    try:
        raw_schema = pa.schema([
            ("user_id", pa.int64()),
            ("seq", pa.list_(pa.struct([
                ("item_id", pa.int64()),
                ("action_type", pa.int32()),
                ("timestamp", pa.int64()),
            ]))),
        ])
        raw_path = temp_path / "raw_sequences.parquet"
        raw_writer = pq.ParquetWriter(raw_path, raw_schema, compression="snappy")
        selected_user_reids = []
        selected_item_reids = set()
        buffer = []
        for row in tqdm(
            _iter_rows(data_path / "seq", ["user_id", "seq"]),
            desc="Selecting TencentGR-1M users",
            total=effective_max_users,
        ):
            if effective_max_users is not None and len(selected_user_reids) >= effective_max_users:
                break
            sequence = row["seq"] or []
            if len(sequence) < 2:
                continue
            selected_user_reids.append(int(row["user_id"]))
            for event in sequence:
                selected_item_reids.add(int(event["item_id"]))
            buffer.append(row)
            if len(buffer) >= 2048:
                _write_rows(raw_writer, buffer, raw_schema)
        _write_rows(raw_writer, buffer, raw_schema)
        raw_writer.close()
        if not selected_user_reids:
            raise ValueError("No valid user sequences were selected")

        item_reids = np.asarray(sorted(selected_item_reids), dtype=np.int64)
        user_reids = np.asarray(selected_user_reids, dtype=np.int64)
        item_to_local = {int(value): index + 1 for index, value in enumerate(item_reids)}
        user_to_local = {int(value): index + 1 for index, value in enumerate(user_reids)}
        feature_cardinalities, reid_to_original = _load_index_metadata(data_path, item_reids)
        item_original_ids = np.asarray([reid_to_original[int(value)] for value in item_reids], dtype=np.int64)
        original_to_local = {
            str(original): index + 1 for index, original in enumerate(item_original_ids.tolist())
        }
        np.save(temp_path / "item_reids.npy", np.pad(item_reids, (1, 0)))
        np.save(temp_path / "item_original_ids.npy", np.pad(item_original_ids, (1, 0)))
        np.save(temp_path / "user_reids.npy", np.pad(user_reids, (1, 0)))

        local_schema = pa.schema([
            ("user_id", pa.int64()),
            ("reindexed_user_id", pa.int64()),
            ("seq", raw_schema.field("seq").type),
        ])
        local_writer = pq.ParquetWriter(temp_path / "sequences.parquet", local_schema, compression="snappy")
        buffer = []
        for row in _iter_rows(raw_path, ["user_id", "seq"]):
            buffer.append({
                "user_id": user_to_local[int(row["user_id"])],
                "reindexed_user_id": int(row["user_id"]),
                "seq": [
                    {
                        "item_id": item_to_local[int(event["item_id"])],
                        "action_type": int(event["action_type"] or 0),
                        "timestamp": int(event["timestamp"] or 0),
                    }
                    for event in row["seq"]
                ],
            })
            if len(buffer) >= 2048:
                _write_rows(local_writer, buffer, local_schema)
        _write_rows(local_writer, buffer, local_schema)
        local_writer.close()
        raw_path.unlink()

        item_feature_columns = list(ITEM_SPARSE)
        item_feature_arrays = {
            feature: np.lib.format.open_memmap(
                temp_path / f"item_feat_{feature}.npy", mode="w+", dtype=np.int64, shape=(len(item_reids) + 1,)
            )
            for feature in ITEM_SPARSE
        }
        for row in tqdm(
            _iter_rows(data_path / "item_feat", ["item_id", *item_feature_columns], batch_size=8192),
            desc="Indexing item features",
        ):
            local_id = item_to_local.get(int(row["item_id"]))
            if local_id is None:
                continue
            for feature in item_feature_columns:
                item_feature_arrays[feature][local_id] = int(row[feature] or 0)
        for array in item_feature_arrays.values():
            array.flush()

        user_columns = [*USER_SPARSE, *USER_ARRAY]
        user_rows = [None] * (len(user_reids) + 1)
        for row in tqdm(
            _iter_rows(data_path / "user_feat", ["user_id", *user_columns], batch_size=8192),
            desc="Indexing user features",
        ):
            local_id = user_to_local.get(int(row["user_id"]))
            if local_id is not None:
                user_rows[local_id] = {"user_id": local_id, **{key: row[key] for key in user_columns}}
        empty_user = {"user_id": 0, **{key: ([] if key in USER_ARRAY else 0) for key in user_columns}}
        user_rows[0] = empty_user
        for local_id in range(1, len(user_rows)):
            if user_rows[local_id] is None:
                user_rows[local_id] = {"user_id": local_id, **{key: ([] if key in USER_ARRAY else 0) for key in user_columns}}
        pq.write_table(pa.Table.from_pylist(user_rows), temp_path / "user_features.parquet", compression="snappy")

        embedding_stats = {}
        for feature_id in mm_emb_ids:
            dimension = EMBEDDING_DIMS[feature_id]
            dtype = np.float32 if feature_id == "81" else np.float16
            embeddings = np.lib.format.open_memmap(
                temp_path / f"emb_{feature_id}.npy",
                mode="w+",
                dtype=dtype,
                shape=(len(item_reids) + 1, dimension),
            )
            present = np.lib.format.open_memmap(
                temp_path / f"emb_{feature_id}_present.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(len(item_reids) + 1,),
            )
            embedding_path = data_path / "mm_emb" / f"emb_{feature_id}_{dimension}_parquet"
            files = _parquet_files(embedding_path)
            for file in tqdm(files, desc=f"Indexing embedding {feature_id}"):
                parquet = pq.ParquetFile(file)
                for batch in parquet.iter_batches(batch_size=4096, columns=["anonymous_cid", "emb"]):
                    ids = batch.column(0).to_pylist()
                    vectors = batch.column(1)
                    for row_index, original_id in enumerate(ids):
                        local_id = original_to_local.get(original_id)
                        if local_id is None or not vectors[row_index].is_valid:
                            continue
                        vector = vectors[row_index].as_py()
                        if len(vector) != dimension:
                            continue
                        embeddings[local_id] = np.asarray(vector, dtype=dtype)
                        present[local_id] = True
            present_indices = np.flatnonzero(present)
            if len(present_indices):
                total = np.zeros(dimension, dtype=np.float64)
                total_squares = np.zeros(dimension, dtype=np.float64)
                for start in range(0, len(present_indices), 8192):
                    indices = present_indices[start:start + 8192]
                    values = np.asarray(embeddings[indices], dtype=np.float64)
                    total += values.sum(axis=0)
                    total_squares += np.square(values).sum(axis=0)
                mean = (total / len(present_indices)).astype(np.float32)
                variance = total_squares / len(present_indices) - np.square(mean.astype(np.float64))
                std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
                std[std < 1e-8] = 1.0
                for start in range(0, len(present_indices), 8192):
                    indices = present_indices[start:start + 8192]
                    embeddings[indices] = ((np.asarray(embeddings[indices], dtype=np.float32) - mean) / std).astype(dtype)
            embeddings.flush()
            present.flush()
            embedding_stats[feature_id] = {
                "dimension": dimension,
                "present": int(present.sum()),
                "missing": int(len(item_reids) - present.sum()),
                "dtype": np.dtype(dtype).name,
            }

        metadata = {
            "format_version": 1,
            "source": str(data_path),
            "profile": profile,
            "max_users": effective_max_users,
            "users": len(user_reids),
            "items": len(item_reids),
            "mm_emb_ids": list(mm_emb_ids),
            "feature_cardinalities": feature_cardinalities,
            "embedding_stats": embedding_stats,
        }
        (temp_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temp_path.rename(cache_path)
        return cache_path
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


class _LazyItemFeatureView:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.itemnum

    def __getitem__(self, index):
        return self.dataset.get_item_feature(index + 1)


class TencentGRDataset(torch.utils.data.Dataset):
    """SASRec-compatible dataset backed by a TencentGR-1M profile cache."""

    def __init__(self, data_dir, args):
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = [str(value) for value in args.mm_emb_id]
        self.cache_path = build_profile_cache(
            data_path=self.data_dir,
            profile=args.profile,
            max_users=args.max_users,
            mm_emb_ids=self.mm_emb_ids,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild_cache,
        )
        self.metadata = json.loads((self.cache_path / "metadata.json").read_text(encoding="utf-8"))
        self.usernum = int(self.metadata["users"])
        self.itemnum = int(self.metadata["items"])
        self.sequences = pq.read_table(self.cache_path / "sequences.parquet").combine_chunks()
        self.user_features = pq.read_table(self.cache_path / "user_features.parquet").combine_chunks()
        self.item_reids = np.load(self.cache_path / "item_reids.npy", mmap_mode="r")
        self.item_original_ids = np.load(self.cache_path / "item_original_ids.npy", mmap_mode="r")
        self.item_feature_arrays = {
            feature: np.load(self.cache_path / f"item_feat_{feature}.npy", mmap_mode="r")
            for feature in ITEM_SPARSE
        }
        self.mm_embeddings = {
            feature: np.load(self.cache_path / f"emb_{feature}.npy", mmap_mode="r")
            for feature in self.mm_emb_ids
        }
        self.feature_types = {
            "user_sparse": USER_SPARSE,
            "item_sparse": ITEM_SPARSE,
            "item_array": [],
            "user_array": USER_ARRAY,
            "item_emb": self.mm_emb_ids,
            "user_continual": [],
            "item_continual": [],
        }
        self.feat_statistics = {
            feature: int(self.metadata["feature_cardinalities"].get(feature, 0))
            for feature in [*USER_SPARSE, *USER_ARRAY, *ITEM_SPARSE]
        }
        self.feature_default_value = {
            **{feature: 0 for feature in [*USER_SPARSE, *ITEM_SPARSE]},
            **{feature: [0] for feature in USER_ARRAY},
            **{feature: np.zeros(EMBEDDING_DIMS[feature], dtype=np.float32) for feature in self.mm_emb_ids},
        }
        self.item_feature_view = _LazyItemFeatureView(self)

    def __len__(self):
        return self.usernum

    def get_item_feature(self, local_item_id):
        feature = self.feature_default_value.copy()
        feature.update({
            key: int(values[local_item_id]) for key, values in self.item_feature_arrays.items()
        })
        for key, embeddings in self.mm_embeddings.items():
            feature[key] = np.asarray(embeddings[local_item_id], dtype=np.float32)
        return feature

    def _get_user_feature(self, local_user_id):
        feature = self.feature_default_value.copy()
        for key in USER_SPARSE:
            value = self.user_features[key][local_user_id].as_py()
            feature[key] = int(value or 0)
        for key in USER_ARRAY:
            value = self.user_features[key][local_user_id].as_py()
            feature[key] = value or [0]
        return feature

    def _random_negative(self, seen):
        if len(seen) >= self.itemnum:
            return int(np.random.randint(1, self.itemnum + 1))
        while True:
            item_id = int(np.random.randint(1, self.itemnum + 1))
            if item_id not in seen:
                return item_id

    def __getitem__(self, index):
        local_user_id = int(self.sequences["user_id"][index].as_py())
        events = self.sequences["seq"][index].as_py()
        extended = [(local_user_id, self._get_user_feature(local_user_id), 2, 0)]
        extended.extend(
            (
                int(event["item_id"]),
                self.get_item_feature(int(event["item_id"])),
                1,
                int(event["action_type"] or 0),
            )
            for event in events
        )
        seq = np.zeros(self.maxlen + 1, dtype=np.int32)
        pos = np.zeros(self.maxlen + 1, dtype=np.int32)
        neg = np.zeros(self.maxlen + 1, dtype=np.int32)
        token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        next_token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        next_action_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        seq_feat = np.empty(self.maxlen + 1, dtype=object)
        pos_feat = np.empty(self.maxlen + 1, dtype=object)
        neg_feat = np.empty(self.maxlen + 1, dtype=object)
        seq_feat[:] = self.feature_default_value
        pos_feat[:] = self.feature_default_value
        neg_feat[:] = self.feature_default_value
        seen = {item_id for item_id, _, kind, _ in extended if kind == 1}
        next_record = extended[-1]
        position = self.maxlen
        for record in reversed(extended[:-1]):
            item_id, feature, kind, _ = record
            next_id, next_feature, next_kind, next_action = next_record
            seq[position] = item_id
            token_type[position] = kind
            next_token_type[position] = next_kind
            next_action_type[position] = next_action
            seq_feat[position] = feature
            if next_kind == 1:
                pos[position] = next_id
                pos_feat[position] = next_feature
                negative_id = self._random_negative(seen)
                neg[position] = negative_id
                neg_feat[position] = self.get_item_feature(negative_id)
            next_record = record
            position -= 1
            if position < 0:
                break
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    @staticmethod
    def collate_fn(batch):
        numeric = [torch.from_numpy(np.asarray(values)) for values in zip(*(row[:6] for row in batch))]
        seq_feat, pos_feat, neg_feat = zip(*(row[6:] for row in batch))
        return (*numeric, list(seq_feat), list(pos_feat), list(neg_feat))


MyDataset = TencentGRDataset
