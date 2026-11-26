"""Tests for bge-reranker reordering and hybrid+rerank retrieval.

The real model (BAAI/bge-reranker-base, ~1.1 GB) is never loaded here:
a fake reranker stands in for it. Real-model behaviour is covered by
the eval (eval/results_100_rerank.json), not by unit tests.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import BM25, HybridRetriever, VectorStore, rerank


class FakeReranker:
    """Scores a chunk 1.0 if any word of the query occurs in it, else 0.0.

    Deterministic and order-reversing when the relevant chunk is last.
    """

    def rerank(self, query, candidates):
        words = query.split()
        scored = [(d, 1.0 if any(w in d["text"] for w in words) else 0.0) for d, _ in candidates]
        scored.sort(key=lambda kv: -kv[1])
        return scored


class TestRerank(unittest.TestCase):
    DOCS = [
        {"doc_id": "a.md", "chunk_id": "a.md#0", "text": "今天天气晴朗适合出门散步"},
        {"doc_id": "b.md", "chunk_id": "b.md#0", "text": "检索增强生成 RAG 先检索文档再生成答案"},
    ]

    def test_reorders_by_reranker_score(self):
        cands = [(self.DOCS[0], 0.9), (self.DOCS[1], 0.1)]
        out = rerank("检索增强生成", cands, reranker=FakeReranker())
        self.assertEqual(out[0][0]["doc_id"], "b.md")
        self.assertEqual(out[0][1], 1.0)

    def test_empty_candidates_returns_empty_without_loading_model(self):
        self.assertEqual(rerank("什么", [], reranker=FakeReranker()), [])


class TestHybridRerank(unittest.TestCase):
    DOCS = [
        {"doc_id": "a.md", "chunk_id": "a.md#0", "text": "检索增强生成 RAG 先检索文档再生成答案"},
        {"doc_id": "b.md", "chunk_id": "b.md#0", "text": "今天天气晴朗适合出门散步"},
    ]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = VectorStore(self.DOCS, Path(self._tmp.name) / "chroma")
        self.bm25 = BM25(self.DOCS)
        self.hybrid = HybridRetriever(self.bm25, self.store, self.DOCS)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reranked_search_answers_fact_question(self):
        hits = self.hybrid.search_with_rerank(
            "什么是 RAG 检索增强生成", top_k=1, candidate_k=2, reranker=FakeReranker())
        self.assertTrue(hits)
        self.assertEqual(hits[0][0]["doc_id"], "a.md")

    def test_refusal_gate_still_applies_before_reranking(self):
        # The gate runs on the fused top-1 before reranking, so reranking
        # can never revive a refused question.
        self.assertEqual(self.hybrid.search_with_rerank(
            "如何用 LoRA 微调一个开源小模型？", top_k=2, candidate_k=2,
            reranker=FakeReranker()), [])
        self.assertEqual(self.hybrid.search_with_rerank(
            "量子计算机超导芯片", top_k=2, candidate_k=2,
            reranker=FakeReranker()), [])

    def test_top_k_respected(self):
        hits = self.hybrid.search_with_rerank(
            "什么是 RAG 检索增强生成", top_k=1, candidate_k=2, reranker=FakeReranker())
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
