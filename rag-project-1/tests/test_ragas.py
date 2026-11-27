import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ragas import (
    context_precision,
    context_recall,
    hit_at_k,
    max_recall_at_k,
    reciprocal_rank,
)

R = "rel.md"   # the expected doc
X = "other.md"  # an irrelevant doc


class TestContextPrecision(unittest.TestCase):
    def test_all_relevant_is_one(self):
        self.assertAlmostEqual(context_precision([R, R, R], R), 1.0)

    def test_no_relevant_is_zero(self):
        self.assertAlmostEqual(context_precision([X, X, X], R), 0.0)
        self.assertAlmostEqual(context_precision([], R), 0.0)

    def test_rank_one_relevant_beats_rank_three(self):
        # [R, X, X]: P@1=1, weighted sum 1, 1 relevant -> 1.0
        self.assertAlmostEqual(context_precision([R, X, X], R), 1.0)
        # [X, X, R]: P@3=1/3, weighted sum 1/3, 1 relevant -> 1/3
        self.assertAlmostEqual(context_precision([X, X, R], R), 1 / 3)

    def test_weighted_formula_hand_computed(self):
        # [R, X, R]: P@1=1/1=1, P@2=1/2, P@3=2/3.
        # weighted sum = 1*1 + 0*(1/2) + 1*(2/3) = 5/3; relevant in list = 2.
        # CP = (5/3) / 2 = 5/6.
        self.assertAlmostEqual(context_precision([R, X, R], R), 5 / 6)

    def test_middle_relevant(self):
        # [X, R, X]: P@2=1/2, weighted sum 1/2, 1 relevant -> 1/2.
        self.assertAlmostEqual(context_precision([X, R, X], R), 0.5)

    def test_two_relevant_late(self):
        # [X, R, R]: P@2=1/2, P@3=2/3; sum = 1/2 + 2/3 = 7/6; /2 = 7/12.
        self.assertAlmostEqual(context_precision([X, R, R], R), 7 / 12)


class TestContextRecall(unittest.TestCase):
    def test_full_coverage(self):
        self.assertAlmostEqual(context_recall([R, R], R, 2), 1.0)

    def test_partial_coverage(self):
        # Doc has 4 chunks, top-3 retrieved 2 of them -> 2/4.
        self.assertAlmostEqual(context_recall([R, X, R], R, 4), 0.5)

    def test_large_doc_ceiling(self):
        # 16-chunk doc, 1 relevant chunk in top-3 -> 1/16.
        self.assertAlmostEqual(context_recall([R, X, X], R, 16), 1 / 16)

    def test_none_retrieved_is_zero(self):
        self.assertAlmostEqual(context_recall([X, X, X], R, 4), 0.0)

    def test_refusal_item_returns_none(self):
        # No ground-truth doc: the caller must exclude the item, not score 0.
        self.assertIsNone(context_recall([], "", 4))

    def test_max_recall_ceiling(self):
        self.assertAlmostEqual(max_recall_at_k(2), 1.0)   # min(3,2)/2
        self.assertAlmostEqual(max_recall_at_k(4), 0.75)  # min(3,4)/4
        self.assertAlmostEqual(max_recall_at_k(16), 3 / 16)


class TestMRRAndHit(unittest.TestCase):
    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank([R, X, X], R), 1.0)
        self.assertAlmostEqual(reciprocal_rank([X, R, X], R), 0.5)
        self.assertAlmostEqual(reciprocal_rank([X, X, R], R), 1 / 3)
        self.assertAlmostEqual(reciprocal_rank([X, X, X], R), 0.0)

    def test_hit_at_k(self):
        self.assertTrue(hit_at_k([R, X, X], R, 1))
        self.assertFalse(hit_at_k([X, R, X], R, 1))
        self.assertTrue(hit_at_k([X, R, X], R, 3))
        self.assertFalse(hit_at_k([X, X, X], R, 3))
        self.assertFalse(hit_at_k([], R, 3))


if __name__ == "__main__":
    unittest.main()
