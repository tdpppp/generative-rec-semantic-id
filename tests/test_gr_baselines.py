import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gr.custom_dataset import ProfileSequenceDataset
from gr.evaluate_gr_baselines import (
    build_most_popular_candidates,
    evaluate_candidate_ranking,
    sample_random_candidates,
)


def semantic_id(index):
    return f"<a_{index}><b_{index}><c_{index}>"


class MetadataDataset:
    split = "test"

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def example_metadata(self, index):
        return self.rows[index]


class GrBaselineTests(unittest.TestCase):
    def test_popularity_excludes_validation_and_test_holdouts(self):
        mapping = {str(index): semantic_id(index) for index in range(1, 7)}
        sequences = [
            [1, 2, 5, 6],
            [2, 3, 5, 6],
        ]
        candidates = build_most_popular_candidates(sequences, mapping, top_k=3)
        self.assertEqual(candidates[0], semantic_id(2))
        self.assertNotIn(semantic_id(5), candidates)
        self.assertNotIn(semantic_id(6), candidates)

    def test_random_candidates_are_reproducible_unique_and_legal(self):
        catalog = [semantic_id(index) for index in range(20)]
        first = sample_random_candidates(catalog, 10, random.Random(2025))
        second = sample_random_candidates(catalog, 10, random.Random(2025))
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(set(first).issubset(catalog))

    def test_metadata_only_profile_does_not_require_tokenizer(self):
        mapping = {str(index * 10): semantic_id(index) for index in range(1, 6)}
        args = SimpleNamespace(padding_side="right", max_seq_length=10, token_depth=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = Path(temp_dir)
            np.save(profile / "item_reids.npy", np.array([0, 10, 20, 30, 40, 50]))
            pq.write_table(
                pa.Table.from_pylist([{
                    "reindexed_user_id": 7,
                    "seq": [
                        {"item_id": item_id, "action_type": 1, "timestamp": item_id}
                        for item_id in (1, 2, 3, 4, 5)
                    ],
                }]),
                profile / "sequences.parquet",
            )
            dataset = ProfileSequenceDataset(profile, mapping, None, args, split="test")
            self.assertEqual(dataset.example_metadata(0)["target_item_id"], 50)
            with self.assertRaisesRegex(RuntimeError, "requires a tokenizer"):
                dataset[0]

    def test_candidate_evaluation_writes_consistent_metrics(self):
        rows = [
            {
                "user_id": 1,
                "target_item_id": 10,
                "target_semantic_id": semantic_id(1),
            },
            {
                "user_id": 2,
                "target_item_id": 20,
                "target_semantic_id": semantic_id(2),
            },
        ]
        dataset = MetadataDataset(rows)
        code_to_items = defaultdict(list, {
            semantic_id(1): [10, 11],
            semantic_id(2): [20],
            semantic_id(3): [30],
        })
        candidates = [semantic_id(1), semantic_id(3)]
        with tempfile.TemporaryDirectory() as temp_dir:
            predictions = Path(temp_dir) / "predictions.jsonl"
            result = evaluate_candidate_ranking(
                "test",
                dataset,
                lambda _index, _row: candidates,
                code_to_items,
                top_k=2,
                profile_path=temp_dir,
                mapping_dir=temp_dir,
                predictions_path=predictions,
            )
            self.assertEqual(result["metrics"]["hits"], 1)
            self.assertEqual(result["metrics"]["hr"], 0.5)
            self.assertEqual(result["ambiguous_target_rate"], 0.5)
            self.assertEqual(len(predictions.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
