import unittest

from sasrec2.metrics import ranking_metrics


class Sasrec2Tests(unittest.TestCase):
    def test_item_ranking_metrics(self):
        result = ranking_metrics([1, 2], 4, prefix="item_")
        self.assertEqual(result["item_hits"], 2)
        self.assertEqual(result["item_hr"], 0.5)
        self.assertAlmostEqual(result["item_mrr"], 0.375)


if __name__ == "__main__":
    unittest.main()
