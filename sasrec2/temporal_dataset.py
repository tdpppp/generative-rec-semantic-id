import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "sasrec") not in sys.path:
    sys.path.insert(0, str(ROOT / "sasrec"))
from tencentgr_dataset import TencentGRDataset


class TemporalSASRecDataset(Dataset):
    """SASRec samples using the same temporal holdouts as the Qwen evaluator."""

    def __init__(self, data_path, profile_path, args, split):
        self.base = TencentGRDataset(data_path, args)
        self.profile_path = Path(profile_path)
        self.split = split
        self.maxlen = args.maxlen
        self.sequences = []
        self.user_ids = []
        self.coordinates = []
        item_reids = np.load(self.profile_path / "item_reids.npy", mmap_mode="r")
        self.reid_to_local = {int(value): index for index, value in enumerate(self.base.item_reids) if index > 0}
        self.user_reid_to_local = {
            int(row["reindexed_user_id"]): int(row["user_id"])
            for row in self.base.sequences.to_pylist()
        }
        table = pq.read_table(self.profile_path / "sequences.parquet").combine_chunks()
        for row in table.to_pylist():
            sequence = []
            for event in row["seq"] or []:
                local = int(event["item_id"])
                item = int(item_reids[local]) if 0 < local < len(item_reids) else 0
                # Profile cache local IDs are mapped to official IDs; base cache uses
                # the same selected-item order, so resolve by official re-ID.
                base_local = self.reid_to_local.get(item, 0)
                if base_local:
                    sequence.append(base_local)
            if len(sequence) < 3:
                continue
            self.user_ids.append(int(row["reindexed_user_id"]))
            self.sequences.append(sequence)
            if split == "train":
                self.coordinates.extend((len(self.sequences) - 1, target) for target in range(1, len(sequence) - 2))
            else:
                self.coordinates.append((len(self.sequences) - 1, len(sequence) - (2 if split == "validation" else 1)))

    def __len__(self):
        return len(self.coordinates)

    def metadata(self, index):
        user_index, target = self.coordinates[index]
        sequence = self.sequences[user_index]
        return {"user_id": self.user_ids[user_index], "history": sequence[:target], "target": sequence[target]}

    def _sample(self, index, include_target=True):
        user_index, target = self.coordinates[index]
        sequence = self.sequences[user_index]
        history = sequence[:target][-self.maxlen:]
        seq = np.zeros(self.maxlen + 1, dtype=np.int32)
        pos = np.zeros(self.maxlen + 1, dtype=np.int32)
        neg = np.zeros(self.maxlen + 1, dtype=np.int32)
        mask = np.zeros(self.maxlen + 1, dtype=np.int32)
        next_mask = np.zeros(self.maxlen + 1, dtype=np.int32)
        action = np.zeros(self.maxlen + 1, dtype=np.int32)
        seq_feat = np.empty(self.maxlen + 1, dtype=object)
        pos_feat = np.empty(self.maxlen + 1, dtype=object)
        neg_feat = np.empty(self.maxlen + 1, dtype=object)
        seq_feat[:] = self.base.feature_default_value
        pos_feat[:] = self.base.feature_default_value
        neg_feat[:] = self.base.feature_default_value
        user_local = self.user_reid_to_local.get(self.user_ids[user_index], 0)
        position = self.maxlen - len(history) + 1
        seq[position - 1] = user_local
        mask[position - 1] = 2
        seq_feat[position - 1] = self.base._get_user_feature(user_local)
        seen = set(sequence)
        for item in history:
            seq[position] = item
            mask[position] = 1
            seq_feat[position] = self.base.get_item_feature(item)
            position += 1
        if include_target:
            target_item = sequence[target]
            target_position = self.maxlen
            pos[target_position] = target_item
            next_mask[target_position] = 1
            pos_feat[target_position] = self.base.get_item_feature(target_item)
            negative = self.base._random_negative(seen)
            neg[target_position] = negative
            neg_feat[target_position] = self.base.get_item_feature(negative)
        return seq, pos, neg, mask, next_mask, action, seq_feat, pos_feat, neg_feat

    def __getitem__(self, index):
        return self._sample(index)

    @staticmethod
    def collate_fn(batch):
        import torch
        numeric = [torch.from_numpy(np.asarray(values)) for values in zip(*(row[:6] for row in batch))]
        features = list(zip(*(row[6:] for row in batch)))
        return (*numeric, *[list(value) for value in features])
