import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gr.custom_dataset import CustomTrainDataset, ProfileSequenceDataset
from gr.evaluate_gr import SemanticIdConstraint, ranking_metrics, resolve_num_beams
from gr.train_gr import resolve_training_output_paths
from gr.utils import load_item2token_dict, semantic_tokens


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sasrec"))

from tencentgr_dataset import ITEM_SPARSE, USER_ARRAY, USER_SPARSE, TencentGRDataset


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False, padding=None, max_length=None, truncation=False):
        pieces = text.replace("><", ">|<").split("|")
        input_ids = list(range(1, len(pieces) + 1))
        if padding == "max_length":
            input_ids = input_ids[:max_length] + [self.pad_token_id] * max(0, max_length - len(input_ids))
        return {"input_ids": input_ids, "attention_mask": [int(token != 0) for token in input_ids]}


class FakeConstraintTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.ids = {
            "<a_1>": 1,
            "<a_2>": 2,
            "<b_3>": 3,
            "<b_4>": 4,
            "<c_5>": 5,
            "<c_6>": 6,
        }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


class RepositoryTests(unittest.TestCase):
    def test_gr_output_and_tensorboard_paths_are_separated(self):
        output_dir, logging_dir = resolve_training_output_paths(
            Path("outputs/gr/checkpoints"),
            Path("outputs/gr/tensorboard"),
            "qwen25_05b_smoke",
            "qwen25_05b_smoke/runs/Sep23_host",
        )
        self.assertEqual(output_dir, Path("outputs/gr/checkpoints/qwen25_05b_smoke"))
        self.assertEqual(logging_dir, Path("outputs/gr/tensorboard/qwen25_05b_smoke/runs/Sep23_host"))

        absolute_log = Path("/tmp/gr-events")
        _, logging_dir = resolve_training_output_paths(
            Path("outputs/gr/checkpoints"),
            Path("outputs/gr/tensorboard"),
            "qwen25_05b_smoke",
            absolute_log,
        )
        self.assertEqual(logging_dir, absolute_log)

    def test_gr_config_semantic_depth_matches(self):
        with (ROOT / "gr" / "gr_train.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        widths = config["model_args"]["se_id_space_width"].split(",")
        self.assertEqual(len(widths), config["data_args"]["token_depth"])

    def test_semantic_tokens_are_complete_and_unique(self):
        tokens = semantic_tokens("2,3,1")
        self.assertEqual(len(tokens), 8)
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertIn("<a_1>", tokens)
        self.assertIn("<b_2>", tokens)
        self.assertIn("<c_0>", tokens)

    def test_item_mapping_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_file = Path(temp_dir) / "worker_0_output.txt"
            mapping_file.write_text("10\t<a_1><b_2><c_3>\n", encoding="utf-8")
            self.assertEqual(load_item2token_dict(temp_dir)["10"], "<a_1><b_2><c_3>")

    def test_gr_labels_only_include_target_semantic_id(self):
        args = SimpleNamespace(
            token_depth=3,
            padding_side="right",
            max_seq_length=4,
            response_flag=True,
        )
        mapping = {
            "1": "<a_0><b_0><c_0>",
            "2": "<a_1><b_1><c_1>",
            "3": "<a_2><b_2><c_2>",
        }
        dataset = CustomTrainDataset("unused.json", mapping, FakeTokenizer(), args)
        example = dataset._process_sequence([1, 2, 3])
        target_labels = [label for label in example["labels"] if label != -100]
        self.assertEqual(len(target_labels), args.token_depth)

    def test_gr_streams_tencentgr_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        args = SimpleNamespace(
            token_depth=3,
            padding_side="right",
            max_seq_length=4,
            response_flag=True,
            data_format="parquet",
            max_train_samples=1,
        )
        mapping = {"1": "<a_0><b_0><c_0>", "2": "<a_1><b_1><c_1>"}
        with tempfile.TemporaryDirectory() as temp_dir:
            seq_dir = Path(temp_dir) / "seq"
            seq_dir.mkdir()
            table = pa.Table.from_pylist([
                {
                    "user_id": 1,
                    "seq": [
                        {"item_id": 1, "action_type": 0, "timestamp": 1},
                        {"item_id": 2, "action_type": 1, "timestamp": 2},
                    ],
                }
            ])
            pq.write_table(table, seq_dir / "part.parquet")
            dataset = CustomTrainDataset(str(seq_dir), mapping, FakeTokenizer(), args)
            example = next(iter(dataset))
            self.assertEqual(sum(label != -100 for label in example["labels"]), 3)

    def test_gr_profile_temporal_splits_use_official_item_ids_without_leakage(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        args = SimpleNamespace(token_depth=3, padding_side="right", max_seq_length=10)
        mapping = {
            str(item_id): f"<a_{item_id}><b_{item_id}><c_{item_id}>"
            for item_id in (10, 20, 30, 40, 50)
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = Path(temp_dir)
            np.save(profile / "item_reids.npy", np.array([0, 10, 20, 30, 40, 50]))
            pq.write_table(
                pa.Table.from_pylist([{
                    "reindexed_user_id": 99,
                    "seq": [
                        {"item_id": item_id, "action_type": 1, "timestamp": item_id}
                        for item_id in (1, 2, 3, 4, 5)
                    ],
                }]),
                profile / "sequences.parquet",
            )
            train = ProfileSequenceDataset(profile, mapping, FakeTokenizer(), args, split="train")
            validation = ProfileSequenceDataset(profile, mapping, FakeTokenizer(), args, split="validation")
            test = ProfileSequenceDataset(profile, mapping, FakeTokenizer(), args, split="test")

            self.assertEqual(len(train), 2)
            self.assertEqual(train.example_metadata(0)["target_item_id"], 20)
            self.assertEqual(train.example_metadata(1)["target_item_id"], 30)
            self.assertEqual(validation.example_metadata(0)["target_item_id"], 40)
            self.assertEqual(test.example_metadata(0)["target_item_id"], 50)
            self.assertEqual(test.example_metadata(0)["history_item_ids"], [10, 20, 30, 40])
            self.assertEqual(sum(label != -100 for label in test[0]["labels"]), 3)

    def test_semantic_id_constraint_only_allows_existing_codes(self):
        import torch

        constraint = SemanticIdConstraint(
            FakeConstraintTokenizer(),
            {"<a_1><b_3><c_5>", "<a_2><b_4><c_6>"},
            prompt_length=2,
        )
        self.assertEqual(constraint.allowed_tokens(0, torch.tensor([9, 9])), [1, 2])
        self.assertEqual(constraint.allowed_tokens(0, torch.tensor([9, 9, 1])), [3])
        self.assertEqual(constraint.allowed_tokens(0, torch.tensor([9, 9, 1, 3])), [5])
        self.assertEqual(constraint.decode([1, 3, 5]), "<a_1><b_3><c_5>")
        self.assertIsNone(constraint.decode([1, 4, 6]))

    def test_ranking_metrics(self):
        import math

        metrics = ranking_metrics([1, 2], total=4)
        self.assertEqual(metrics["hr"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 0.375)
        self.assertAlmostEqual(metrics["ndcg"], (1 + 1 / math.log2(3)) / 4)

    def test_num_beams_defaults_to_top_k_and_rejects_narrow_search(self):
        self.assertEqual(resolve_num_beams(10), 10)
        self.assertEqual(resolve_num_beams(10, 20), 20)
        with self.assertRaisesRegex(ValueError, "greater than or equal to top_k"):
            resolve_num_beams(10, 9)

    def test_profile_history_limit_keeps_most_recent_items(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        args = SimpleNamespace(token_depth=3, padding_side="right", max_seq_length=2)
        mapping = {
            str(item_id): f"<a_{item_id}><b_{item_id}><c_{item_id}>"
            for item_id in (10, 20, 30, 40, 50)
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = Path(temp_dir)
            np.save(profile / "item_reids.npy", np.array([0, 10, 20, 30, 40, 50]))
            pq.write_table(
                pa.Table.from_pylist([{
                    "reindexed_user_id": 99,
                    "seq": [
                        {"item_id": item_id, "action_type": 1, "timestamp": item_id}
                        for item_id in (1, 2, 3, 4, 5)
                    ],
                }]),
                profile / "sequences.parquet",
            )
            validation = ProfileSequenceDataset(profile, mapping, FakeTokenizer(), args, split="validation")
            self.assertEqual(validation.example_metadata(0)["history_item_ids"], [20, 30])

    def test_tencentgr_profile_uses_compact_ids_and_keeps_reids(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in ("seq", "item_feat", "user_feat", "mm_emb/emb_81_32_parquet"):
                (root / relative).mkdir(parents=True)
            seq_rows = [{
                "user_id": 7,
                "seq": [
                    {"item_id": 10, "action_type": None, "timestamp": 1},
                    {"item_id": 20, "action_type": 1, "timestamp": 2},
                ],
            }]
            pq.write_table(pa.Table.from_pylist(seq_rows), root / "seq/part.parquet")
            item_rows = []
            for item_id in (10, 20):
                item_rows.append({"item_id": item_id, **{key: 1 for key in ITEM_SPARSE}})
            pq.write_table(pa.Table.from_pylist(item_rows), root / "item_feat/part.parquet")
            user_row = {"user_id": 7, **{key: 1 for key in USER_SPARSE}, **{key: [1] for key in USER_ARRAY}}
            pq.write_table(pa.Table.from_pylist([user_row]), root / "user_feat/part.parquet")
            emb_rows = [
                {"anonymous_cid": "1000", "emb": [0.1] * 32},
                {"anonymous_cid": "2000", "emb": [0.2] * 32},
            ]
            pq.write_table(pa.Table.from_pylist(emb_rows), root / "mm_emb/emb_81_32_parquet/part.parquet")
            feature_ids = [*ITEM_SPARSE, *USER_SPARSE, *USER_ARRAY]
            indexer = {
                "f": {key: {1: 1} for key in feature_ids},
                "i": {1000: 10, 2000: 20},
                "u": {"user": 7},
                "a": {0: 1, 1: 2},
            }
            with (root / "indexer.pkl").open("wb") as stream:
                pickle.dump(indexer, stream)
            args = SimpleNamespace(
                maxlen=4,
                mm_emb_id=["81"],
                profile="smoke",
                max_users=1,
                cache_dir=None,
                rebuild_cache=False,
            )
            dataset = TencentGRDataset(root, args)
            self.assertEqual(dataset.itemnum, 2)
            self.assertEqual(dataset.item_reids[1:].tolist(), [10, 20])
            self.assertEqual(dataset.item_original_ids[1:].tolist(), [1000, 2000])
            self.assertEqual(dataset[0][0].shape, (5,))
            self.assertTrue(np.isfinite(dataset.mm_embeddings["81"]).all())

    def test_documented_entrypoints_exist(self):
        for path in (
            "sasrec/main.py",
            "rqvae/train/rqvae_train.py",
            "rqvae/infer/rqvae_infer.py",
            "gr/train_gr.py",
        ):
            self.assertTrue((ROOT / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
