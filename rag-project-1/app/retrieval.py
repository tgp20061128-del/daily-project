"""Retrieval: BM25 with Chinese word segmentation + a refusal gate.

Tokenization
------------
Chinese is segmented with jieba when installed (``pip install jieba``);
without it we fall back to character bigrams (still far better than the
old single-character split, where generic words like "模型" matched
everything and refusal questions scored like real ones).

Stopwords (question words, single-character function words, punctuation)
are dropped before scoring so they cannot inflate a match.

Refusal gate
------------
BM25 scores alone cannot separate "should answer" from "should refuse"
on a small corpus: a refusal question can score 2+ just by sharing the
generic word "模型" with every chunk. So ``search()`` refuses unless
the top-1 chunk shares at least 2 distinct *content* tokens (length >= 2
after segmentation + stopword filtering) with the query -- or all of the
query's content tokens when the query has fewer than 2
(e.g. "什么是 RAG?" has only one: "rag").

A second rule catches near-miss questions whose topic words genuinely
occur in the corpus ("Chroma 向量库的默认端口是多少？" -- the corpus
mentions Chroma as an example vector store, but never states a port):
if >= 2 of the query's content tokens are absent from the *entire*
corpus vocabulary, and the top-1 chunk already covers every in-vocabulary
content token the question mentions, the question's core is outside the
corpus and the hit is refused. Format-instruction words (注明/出处/分点)
and generic quantity words (一个/一次/这个) are stopwords, so they can
neither fake a match nor fake an out-of-vocabulary signal.

This is a lexical heuristic, not a semantic judgement. Its known limit:
on a large corpus two shared content words can still be a coincidence.
The planned upgrade is vector retrieval + bge-reranker as a second
opinion (see README "下一步").

Vector search (Chroma) is wired: ``VectorStore`` persists a collection under
``data/chroma/`` with Chroma's default embedding (all-MiniLM-L6-v2, ONNX).
``HybridRetriever`` fuses BM25 + vector with RRF (k=60) and applies the same
refusal gate to the fused top-1 chunk, so all three modes are compared under
one refusal rule. ``Reranker`` (BAAI/bge-reranker-base cross-encoder, lazy
loaded) reorders hybrid candidates; see ``HybridRetriever.search_with_rerank``.
"""
import math
import os
import re
from collections import Counter

try:
    import jieba
except ImportError:  # stdlib-only fallback: character bigrams
    jieba = None

# Question words, single-character function words, generic quantity /
# demonstrative words, and format-instruction words. Dropped before
# scoring; they carry no topical signal and only inflate spurious matches
# ("一个" once let a refusal question pass by pairing with "模型";
# 注明/出处 are answer-format instructions, not topics). The glued
# pronoun forms (这是/那是/它是/这里/那里) are here because jieba
# sometimes segments them as one token -- they are still pronouns.
STOPWORDS = set(
    "的 是 了 在 和 与 有 个 一 不 也 都 就 还 对 于 以 及 或 但 若 如 之 乎 者 也 呢 吗 吧 啊 哦 嗯 "
    "什么 怎么 如何 为什么 多少 哪 哪里 怎样 几 是否 能 可以 会 要 想 请 给 用 把 被 从 到 向 为 那 它 这 "
    "一个 一次 一些 一种 一点 这个 那个 这些 那些 这样 那样 "
    "这是 那是 它是 这里 那里 "
    "注明 出处 分点 每点 列出 ".split()
)

# Minimum distinct content tokens the top-1 chunk must share with the query.
# 1 shared word ("模型" is in every chunk) proves nothing; 2 is real evidence.
MIN_CONTENT_OVERLAP = 2
# Minimum out-of-vocabulary content tokens for the second refusal rule.
MIN_OOV_REFUSAL = 2

_NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff]+")
_NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")


def _segments(text):
    """Word-level segments: jieba when available, char bigrams otherwise."""
    if jieba is not None:
        return jieba.lcut(text)
    # Fallback: ASCII words as-is, Chinese as overlapping bigrams.
    out = []
    for m in re.finditer(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+|[^\w\s]", text):
        tok = m.group(0)
        if "\u4e00" <= tok[0] <= "\u9fff":
            if len(tok) >= 2:
                out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
            else:
                out.append(tok)
        else:
            out.append(tok)
    return out


def tokenize(text):
    """Lowercased content tokens: segmented, stopwords/punctuation dropped."""
    out = []
    for tok in _segments(text):
        t = tok.strip().lower()
        if not t or _NON_WORD_RE.fullmatch(t):
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out


def content_tokens(text):
    """Tokens that carry topical signal: length >= 2 after tokenization.

    Pure numbers (200, 0.7) are not topical vocabulary: a question asking
    "200、500、1200 字符时召回率各是多少" is about 块大小/召回率, and
    the numerals being absent from the corpus must not count as
    out-of-vocabulary evidence in the refusal gate. Alphanumeric tokens
    such as 7B or bge-small are kept.
    """
    return [t for t in tokenize(text) if len(t) >= 2 and not _NUMERIC_RE.fullmatch(t)]


def rewrite_query(question, history=None):
    """Expand an elliptical follow-up question with content words from history.

    ``history`` is prior question texts, oldest first. Content tokens
    (see ``content_tokens``: jieba-segmented, stopwords removed,
    length >= 2, pure numbers excluded) from history that the current
    question lacks are appended, first-seen order, deduplicated. The
    original wording is never changed -- words are only appended.
    Pronouns (它/这/那) are stopwords, so they can never be added.
    With no history, or when history adds nothing new, the question is
    returned unchanged -- so callers without history behave exactly as
    before. Rule-based only: it restores dropped topic words, it cannot
    resolve complex ellipsis (e.g. "那它呢" with no topic in history);
    that needs an LLM rewrite.
    """
    if not history:
        return question
    current = set(content_tokens(question))
    extra = []
    for h in history:
        for t in content_tokens(h):
            if t not in current:
                extra.append(t)
                current.add(t)
    if not extra:
        return question
    return question + " " + " ".join(extra)


def gate_refuses(tokenized_docs, query, top_idx, vocab=None):
    """True when the top hit is too weak to answer the query.

    Rule 1: requires >= MIN_CONTENT_OVERLAP shared content tokens, or all
    of the query's content tokens when it has fewer. A query with no
    content tokens at all (only stopwords) is always refused.
    Rule 2 (needs ``vocab``, the corpus vocabulary): if >= MIN_OOV_REFUSAL
    content tokens are absent from the whole corpus, and the top chunk
    already covers every in-vocabulary content token, the question's core
    is outside the corpus -- refuse even though the count rule passed.
    Without ``vocab`` only rule 1 applies. Shared by BM25, vector and
    hybrid search so the refusal rule is identical everywhere and the
    comparison between modes measures ranking only.
    """
    q_content = set(content_tokens(query))
    if not q_content:
        return True
    shared = q_content & set(tokenized_docs[top_idx])
    if len(shared) < min(MIN_CONTENT_OVERLAP, len(q_content)):
        return True
    if vocab is not None:
        oov = q_content - set(vocab)
        in_vocab = q_content & set(vocab)
        if len(oov) >= MIN_OOV_REFUSAL and not (in_vocab - shared):
            return True
    return False


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs  # list of {"doc_id": str, "text": str}
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(d["text"]) for d in docs]
        self.doc_len = [len(t) for t in self.tokenized]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        df = Counter()
        for toks in self.tokenized:
            for t in set(toks):
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query):
        q = tokenize(query)
        scores = []
        for i, toks in enumerate(self.tokenized):
            tf = Counter(toks)
            s = 0.0
            dl = self.doc_len[i] or 1
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                )
            scores.append(s)
        return scores

    def _refuse(self, query, top_idx):
        """True when the top hit is too weak to answer the query."""
        return gate_refuses(self.tokenized, query, top_idx, vocab=self.idf)

    def search(self, query, top_k=5, history=None):
        """Top-k [(doc, score)] best first, or [] when the gate refuses.

        ``history`` (prior question texts) expands an elliptical
        follow-up via :func:`rewrite_query` before scoring and gating;
        ``None``/empty behaves exactly as before.
        """
        query = rewrite_query(query, history)
        scores = self.score(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if not ranked or scores[ranked[0]] <= 0:
            return []
        if self._refuse(query, ranked[0]):
            return []
        return [(self.docs[i], scores[i]) for i in ranked[:top_k] if scores[i] > 0]


class Reranker:
    """Cross-encoder reranker backed by BAAI/bge-reranker-base.

    278M params, ~1.1 GB download from HuggingFace (cached under
    ``~/.cache/huggingface/``). Unlike the bi-encoder embedding, which
    scores the query and each chunk independently, the cross-encoder
    scores (query, chunk) pairs jointly, so it can use the interaction
    between the words of the question and the words of the chunk.

    The import lives inside ``__init__`` so BM25-only use stays light,
    and the model is loaded lazily (see ``get_reranker``) so importing
    this module never downloads or loads 1.1 GB. Scores are raw logits
    (higher = more relevant); only the ordering is meaningful.
    """

    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self, model_name=MODEL_NAME, device="cpu"):
        from sentence_transformers import CrossEncoder

        _sanitize_no_proxy_for_httpx()  # huggingface_hub also downloads via httpx
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query, texts, batch_size=32):
        """Cross-encoder scores for [(query, text), ...], one per text."""
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False)]

    def rerank(self, query, candidates):
        """Reorder [(doc, score), ...] by cross-encoder score, best first.

        Returns [(doc, reranker_score), ...]; the returned score replaces
        the input score (RRF or distance) because only the reranker
        ordering is meaningful after this point.
        """
        if not candidates:
            return []
        scores = self.score(query, [d["text"] for d, _ in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [(candidates[i][0], scores[i]) for i in order]


_reranker = None


def get_reranker():
    """Process-wide lazy singleton: loads the 1.1 GB model on first use."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def rerank(query, candidates, reranker=None):
    """Reorder [(doc, score), ...] with the bge cross-encoder.

    Pass a ``reranker`` (any object with ``.rerank(query, candidates)``)
    in tests to avoid loading the real model; otherwise the lazy
    singleton is used. Empty input returns [] without loading anything.
    """
    if not candidates:
        return []
    r = reranker if reranker is not None else get_reranker()
    return r.rerank(query, candidates)


def _sanitize_no_proxy_for_httpx():
    """Drop NO_PROXY entries httpx cannot parse.

    Chroma downloads its default embedding model with httpx, and httpx
    chokes on this sandbox's NO_PROXY: unbracketed IPv6 literals such as
    ``fd8b:...::1`` are read as host:port and raise ``InvalidURL``. Keep
    only entries httpx parses (plain hosts, ``::1``). Process-local only;
    external downloads still go through the proxy. No-op once the model
    is cached (``~/.cache/chroma/onnx_models/``).
    """
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key)
        if not raw:
            continue
        kept = [e for e in (x.strip() for x in raw.split(",")) if ":" not in e or e in ("::1", "[::1]")]
        os.environ[key] = ",".join(kept)


class BGEEmbeddingFunction:
    """Chroma embedding function backed by BAAI/bge-small-zh-v1.5.

    Chinese retrieval model (512-dim, ~94 MB download from HuggingFace).
    Requires sentence-transformers + torch (heavy: torch CPU wheel alone
    is ~190 MB; the PyPI CUDA build is far larger -- install CPU-only
    torch first, see README). The import lives inside ``__init__`` so
    BM25-only use and the default ONNX embedding stay light. Embeddings
    are L2-normalized, so Chroma's default squared-L2 ranking equals
    cosine ranking.

    Registered as a Chroma ``EmbeddingFunction`` so the protocol's
    ``embed_query`` exists: Chroma 1.x calls ``embed_query`` for queries
    and ``__call__`` for documents, which lets bge's query instruction
    (``为这个句子生成表示以用于检索相关文章：``) apply to queries only --
    the correct bge usage.
    """

    MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name=MODEL_NAME):
        from chromadb import EmbeddingFunction
        from sentence_transformers import SentenceTransformer

        _sanitize_no_proxy_for_httpx()  # huggingface_hub also downloads via httpx
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        EmbeddingFunction.register(self.__class__)

    def _encode(self, input, query=False):
        texts = [f"{self.QUERY_INSTRUCTION}{t}" if query else t for t in input]
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def __call__(self, input):
        return self._encode(input, query=False)

    def embed_query(self, input):
        return self._encode(input, query=True)

    def embed_documents(self, input):
        return self._encode(input, query=False)

    def name(self):
        return "bge-small-zh-v1.5"

    def get_config(self):
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config):
        return BGEEmbeddingFunction(model_name=config.get("model_name", BGEEmbeddingFunction.MODEL_NAME))


def default_embedding_function(name="default"):
    """Build an embedding function by name; also returns its collection suffix.

    - "default": Chroma's DefaultEmbeddingFunction (all-MiniLM-L6-v2, ONNX).
    - "bge-small-zh-v1.5": BGEEmbeddingFunction (Chinese, needs torch).
    """
    if name == "default":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction()
    if name == "bge-small-zh-v1.5":
        return BGEEmbeddingFunction()
    raise ValueError(f"unknown embedding: {name!r} (expected 'default' or 'bge-small-zh-v1.5')")


class VectorStore:
    """Chroma persistent vector store.

    Default embedding is Chroma's ``DefaultEmbeddingFunction``
    (all-MiniLM-L6-v2, ONNX, 384-dim; ~79 MB download on first use,
    cached under ``~/.cache/chroma/onnx_models/``). The collection lives
    under ``persist_dir`` (project ``data/chroma/``) and is rebuilt
    whenever the chunk ids change, so re-running ingest stays in sync.

    ``search`` returns raw nearest neighbours ordered by ascending
    distance (Chroma's default space is squared L2) with NO refusal
    gate: the gate is a policy applied by the caller, so BM25, vector
    and hybrid are compared under the same rule.
    """

    COLLECTION_NAME = "rag-chunks"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Chroma DefaultEmbeddingFunction (ONNX)

    def __init__(self, docs, persist_dir, collection_name=None, embedding_function=None):
        import chromadb

        _sanitize_no_proxy_for_httpx()
        self.docs = list(docs)
        self.ids = [d["chunk_id"] for d in self.docs]
        self.index = {cid: i for i, cid in enumerate(self.ids)}
        client = chromadb.PersistentClient(path=str(persist_dir))
        ef = embedding_function or default_embedding_function("default")
        ef_name = ef.name() if hasattr(ef, "name") else "default"
        if collection_name is None:
            collection_name = self.COLLECTION_NAME + ("" if ef_name == "default" else f"-{ef_name}")
        self.collection_name = collection_name
        self.embedding_name = ef_name
        col = client.get_or_create_collection(name=collection_name, embedding_function=ef)
        existing = set(col.get(ids=self.ids)["ids"]) if col.count() else set()
        if col.count() != len(self.docs) or existing != set(self.ids):
            client.delete_collection(name=collection_name)
            col = client.create_collection(name=collection_name, embedding_function=ef)
            col.add(
                ids=self.ids,
                documents=[d["text"] for d in self.docs],
                metadatas=[{"doc_id": d["doc_id"], "chunk_id": d["chunk_id"]} for d in self.docs],
            )
        self._collection = col

    def search_indices(self, query, top_k=5, history=None):
        """Full ranking as [(chunk_index, distance)] ascending by distance."""
        query = rewrite_query(query, history)
        n = min(top_k, len(self.docs))
        if n <= 0:
            return []
        res = self._collection.query(query_texts=[query], n_results=n)
        return [(self.index[cid], float(dist)) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def search(self, query, top_k=5, history=None):
        """[(doc, distance)] ranked nearest first. No refusal gate."""
        return [(self.docs[i], dist) for i, dist in self.search_indices(query, top_k, history=history)]


def rrf_fuse(rankings, rrf_k=60):
    """Reciprocal Rank Fusion over lists of chunk indices (best first).

    Score of a chunk = sum over lists of 1 / (rrf_k + rank), rank 1-based.
    Chunks absent from a list contribute 0 from that list. Returns
    [(chunk_idx, rrf_score)] sorted by score desc, ties by chunk index.
    """
    fused = Counter()
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """BM25 + vector, fused with RRF (k=60).

    The same refusal gate as BM25 is applied to the fused top-1 chunk,
    so refusal behaviour is identical across modes. Scores returned are
    RRF scores (not BM25 scores); only the ranking is meaningful.
    """

    def __init__(self, bm25, vector_store, docs, rrf_k=60):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.docs = docs
        self.rrf_k = rrf_k

    def _fused_ranking(self, query):
        """Full RRF ranking as [(chunk_idx, rrf_score)] best first."""
        bm25_ranked = [i for i, s in sorted(enumerate(self.bm25.score(query)), key=lambda kv: -kv[1]) if s > 0]
        vec_ranked = [i for i, _ in self.vector_store.search_indices(query, top_k=len(self.docs))]
        return rrf_fuse([bm25_ranked, vec_ranked], self.rrf_k)

    def search(self, query, top_k=5, history=None):
        query = rewrite_query(query, history)
        fused = self._fused_ranking(query)
        if not fused:
            return []
        if gate_refuses(self.bm25.tokenized, query, fused[0][0], vocab=self.bm25.idf):
            return []
        return [(self.docs[i], round(s, 4)) for i, s in fused[:top_k]]

    def search_with_rerank(self, query, top_k=3, candidate_k=10, reranker=None, history=None):
        """Hybrid RRF top-``candidate_k``, then cross-encoder rerank to top_k.

        The refusal gate is applied to the fused top-1 chunk BEFORE
        reranking, exactly as in ``search()``: reranking can only reorder
        candidates, never revive a refused question. Pass a fake
        ``reranker`` in tests to avoid loading the real model.
        ``history`` expands the query (see :func:`rewrite_query`) and the
        expanded query is used for fusion, gating and reranker scoring.
        """
        query = rewrite_query(query, history)
        fused = self._fused_ranking(query)
        if not fused:
            return []
        if gate_refuses(self.bm25.tokenized, query, fused[0][0], vocab=self.bm25.idf):
            return []
        candidates = [(self.docs[i], round(s, 4)) for i, s in fused[:candidate_k]]
        return rerank(query, candidates, reranker=reranker)[:top_k]
