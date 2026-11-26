import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunk import chunk_text
from app.retrieval import BM25, content_tokens, gate_refuses, tokenize


class TestChunk(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(len(chunk_text("hello world")), 1)

    def test_long_block_splits(self):
        text = "字" * 1200
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"doc_id": "a.md", "text": "检索增强生成 RAG 先检索文档再生成答案"},
            {"doc_id": "b.md", "text": "今天天气晴朗适合出门散步"},
        ]
        self.bm25 = BM25(self.docs)

    def test_retrieves_relevant_doc_first(self):
        hits = self.bm25.search("什么是 RAG 检索增强生成", top_k=2)
        self.assertEqual(hits[0][0]["doc_id"], "a.md")

    def test_no_match_returns_empty(self):
        hits = self.bm25.search("量子计算机超导芯片", top_k=2)
        self.assertEqual(hits, [])

    def test_refuses_out_of_corpus_question(self):
        # Shares only the generic word "模型" with the corpus: 1 content
        # token < MIN_CONTENT_OVERLAP, so it must refuse, not answer.
        hits = self.bm25.search("如何用 LoRA 微调一个开源小模型？", top_k=2)
        self.assertEqual(hits, [])

    def test_single_content_token_question_still_answers(self):
        # "什么是 RAG?" has one content token ("rag"); sharing all of it
        # is enough -- the gate is min(2, len(query content tokens)).
        hits = self.bm25.search("什么是 RAG", top_k=2)
        self.assertEqual(hits[0][0]["doc_id"], "a.md")


class TestRefusalGate(unittest.TestCase):
    def test_generic_and_instruction_words_are_stopwords(self):
        # "一个" once let a refusal pass by pairing with "模型"; 注明/出处
        # are answer-format instructions, not topics.
        self.assertNotIn("一个", tokenize("这是一个模型"))
        self.assertNotIn("这个", tokenize("这个模型怎么样"))
        self.assertEqual(tokenize("注明出处"), [])

    def test_pure_numbers_are_not_content_tokens(self):
        # "200、500、1200 字符时召回率各是多少" is about 块大小/召回率;
        # the numerals must not count as out-of-vocabulary evidence.
        toks = content_tokens("块大小 200、500、1200 字符时召回率各是多少")
        self.assertNotIn("200", toks)
        self.assertNotIn("500", toks)
        self.assertNotIn("1200", toks)
        self.assertIn("召回", toks)

    def test_rule2_refuses_when_core_is_out_of_corpus(self):
        # Top chunk shares 2 tokens (chroma, 向量) so rule 1 passes, but
        # 端口/默认 are absent from the whole corpus and the chunk already
        # covers every in-vocabulary token: the core is outside the corpus.
        tokenized = [["chroma", "向量", "库"], ["天气", "晴朗"]]
        vocab = {"chroma", "向量", "库", "天气", "晴朗"}
        self.assertTrue(gate_refuses(
            tokenized, "Chroma 向量库的默认端口是多少？", 0, vocab=vocab))

    def test_rule2_does_not_fire_when_a_vocab_token_is_unmatched(self):
        # Same OOV count, but 向量 is in the corpus yet missing from the
        # top chunk: the top chunk may just be the wrong chunk, so the
        # gate must not refuse on rule 2.
        tokenized = [["单纯", "检索"], ["向量", "检索"]]
        vocab = {"单纯", "向量", "检索", "混合"}
        self.assertFalse(gate_refuses(
            tokenized, "那它和单纯向量检索比有什么优势？", 0, vocab=vocab))

    def test_rule2_needs_vocab(self):
        # Without a vocabulary only rule 1 applies: 2 shared tokens pass.
        tokenized = [["chroma", "向量", "库"], ["天气", "晴朗"]]
        self.assertFalse(gate_refuses(
            tokenized, "Chroma 向量库的默认端口是多少？", 0))


if __name__ == "__main__":
    unittest.main()
