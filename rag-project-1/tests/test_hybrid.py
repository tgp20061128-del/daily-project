"""Tests for Chroma vector retrieval and RRF hybrid retrieval.

Uses the default (ONNX) embedding -- the model is cached after the first
run, so these tests need network only once. bge is covered by the eval
(eval/results_bge.json), not by unit tests: loading torch takes ~30 s.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import BM25, HybridRetriever, VectorStore, rrf_fuse


class TestRRF(unittest.TestCase):
    def test_fuses_two_rankings(self):
        fused = rrf_fuse([[0, 1, 2], [1, 0, 2]], rrf_k=60)
        # Chunks 0 and 1 each rank 1st once and 2nd once: tie, index order.
        self.assertEqual([i for i, _ in fused], [0, 1, 2])
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62, places=6)

    def test_chunk_missing_from_one_list_still_ranks(self):
        fused = rrf_fuse([[0], [1]], rrf_k=60)
        self.assertEqual([i for i, _ in fused], [0, 1])


class VectorStoreTestBase(unittest.TestCase):
    DOCS = [
        {"doc_id": "a.md", "chunk_id": "a.md#0", "text": "检索增强生成 RAG 先检索文档再生成答案"},
        {"doc_id": "b.md", "chunk_id": "b.md#0", "text": "今天天气晴朗适合出门散步"},
    ]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = VectorStore(self.DOCS, Path(self._tmp.name) / "chroma")

    def tearDown(self):
        self._tmp.cleanup()


class TestVectorStore(VectorStoreTestBase):
    def test_recalls_relevant_chunk(self):
        hits = self.store.search("什么是 RAG 检索增强生成", top_k=2)
        self.assertEqual(hits[0][0]["doc_id"], "a.md")

    def test_distances_ascending(self):
        hits = self.store.search("什么是 RAG 检索增强生成", top_k=2)
        self.assertLessEqual(hits[0][1], hits[1][1])


class TestHybrid(VectorStoreTestBase):
    def setUp(self):
        super().setUp()
        self.bm25 = BM25(self.DOCS)
        self.hybrid = HybridRetriever(self.bm25, self.store, self.DOCS)

    def test_answers_fact_question(self):
        hits = self.hybrid.search("什么是 RAG 检索增强生成", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0]["doc_id"], "a.md")

    def test_refusal_gate_survives_fusion(self):
        # Out-of-corpus questions share < 2 content tokens with any chunk,
        # so the fused top-1 is refused exactly like BM25 refuses it.
        self.assertEqual(self.hybrid.search("如何用 LoRA 微调一个开源小模型？", top_k=2), [])
        self.assertEqual(self.hybrid.search("量子计算机超导芯片", top_k=2), [])


if __name__ == "__main__":
    unittest.main()
