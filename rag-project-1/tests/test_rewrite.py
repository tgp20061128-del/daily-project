import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import BM25, rewrite_query


class TestRewriteQuery(unittest.TestCase):
    def test_adds_missing_content_words_from_history(self):
        q = rewrite_query("那它和单纯向量检索比有什么优势？", ["混合检索为什么比单一检索好？"])
        self.assertTrue(q.startswith("那它和单纯向量检索比有什么优势？"))
        for w in ("混合", "检索", "单一"):
            self.assertIn(w, q)

    def test_dedupes_words_already_in_question(self):
        # "检索" is in both the question and the history: appended once at most.
        q = rewrite_query("检索没召回怎么办？", ["混合检索为什么比单一检索好？"])
        self.assertEqual(q.count("检索"), 1)

    def test_pronouns_are_never_added(self):
        # 它/这/那 are stopwords: a history of only pronouns adds nothing.
        self.assertEqual(rewrite_query("那它呢？", ["它是什么？"]), "那它呢？")
        self.assertEqual(rewrite_query("这呢？", ["这是什么？", "那又是什么？"]), "这呢？")

    def test_no_history_returns_question_unchanged(self):
        self.assertEqual(rewrite_query("什么是 RAG？", None), "什么是 RAG？")
        self.assertEqual(rewrite_query("什么是 RAG？", []), "什么是 RAG？")

    def test_history_with_nothing_new_returns_unchanged(self):
        self.assertEqual(
            rewrite_query("什么是 RAG？", ["什么是 RAG？"]),
            "什么是 RAG？",
        )

    def test_history_order_is_first_seen(self):
        q = rewrite_query("多少？", ["混合检索比单一检索好", "向量数据库查询快"])
        extra = q.split(" ", 1)[1].split(" ")
        self.assertEqual(extra, ["混合", "检索", "单一", "向量", "数据库", "查询"])


class TestRewriteRetrieval(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"doc_id": "a.md", "text": "混合检索的融合召回率达到百分之九十，融合方法结合 BM25 与向量检索"},
            {"doc_id": "b.md", "text": "单纯向量检索的召回率是百分之八十，速度很快"},
        ]
        self.bm25 = BM25(self.docs)

    def test_history_flips_ranking_to_the_right_doc(self):
        # Bare follow-up points at the wrong doc (b: 单纯向量检索).
        bare = self.bm25.search("那它比单纯向量检索高多少？", top_k=2)
        self.assertEqual(bare[0][0]["doc_id"], "b.md")
        # With the previous question's topic words, the right doc (a) wins.
        with_history = self.bm25.search(
            "那它比单纯向量检索高多少？", top_k=2,
            history=["单独用向量检索和 BM25 的召回率各是多少？混合后是多少？"],
        )
        self.assertEqual(with_history[0][0]["doc_id"], "a.md")

    def test_no_history_behaves_as_before(self):
        hits = self.bm25.search("量子计算机超导芯片", top_k=2, history=[])
        self.assertEqual(hits, [])
        hits = self.bm25.search("量子计算机超导芯片", top_k=2)
        self.assertEqual(hits, [])


class TestAskHistory(unittest.TestCase):
    def test_ask_accepts_history_and_uses_it(self):
        from fastapi.testclient import TestClient

        from app.main import app

        c = TestClient(app)
        r = c.post("/ask", json={
            "question": "那它和单纯向量检索比有什么优势？",
            "mode": "bm25",
            "history": ["混合检索为什么比单一检索好？"],
        })
        self.assertFalse(r.json()["refused"])
        # Top-3 hit rule (same as eval): the parent question's doc must be
        # among the citations. Top-1 may legitimately be the deeper 03 doc.
        self.assertIn("01-rag-intro.md", [cit["doc_id"] for cit in r.json()["citations"]])
        # No history at all: same shape as before, mode still works.
        r2 = c.post("/ask", json={"question": "什么是 RAG？", "mode": "bm25"})
        self.assertFalse(r2.json()["refused"])


if __name__ == "__main__":
    unittest.main()
