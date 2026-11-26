"""FastAPI service. Retrieval only for now: /ask returns cited chunks,
refuses when nothing relevant is found. LLM answer generation is TODO
(see README) — no answer is invented when retrieval finds nothing.

Retrieval modes (query param `mode`, default "bm25" -- behaviour
unchanged from before hybrid was added):
- bm25:   BM25 (jieba) + refusal gate.
- vector: Chroma vector search (default embedding) + same gate.
- hybrid: BM25 + vector fused with RRF (k=60) + same gate.
- hybrid_rerank: hybrid top-10 candidates reranked by the bge
  cross-encoder to top_k (gate applied before reranking, so a
  refused question stays refused). Loads the 1.1 GB model on
  first use; the other modes never touch it.
"""
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.ingest import build_index
from app.retrieval import BM25, HybridRetriever, VectorStore, gate_refuses, get_reranker, rewrite_query

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

app = FastAPI(title="RAG Project 1")
_bm25 = None
_vector_store = None
_hybrid = None


def get_bm25():
    global _bm25
    if _bm25 is None:
        _bm25 = BM25(build_index())
    return _bm25


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(build_index(), CHROMA_DIR)
    return _vector_store


def get_hybrid():
    global _hybrid
    if _hybrid is None:
        bm25 = get_bm25()
        _hybrid = HybridRetriever(bm25, get_vector_store(), bm25.docs)
    return _hybrid


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "bm25"
    # Prior turns (oldest first) for multi-turn follow-ups. When present,
    # the question is expanded with content words from history before
    # retrieval (see app.retrieval.rewrite_query); omitted/empty behaves
    # exactly as before.
    history: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    mode = req.mode.lower()
    history = req.history or []
    if mode == "bm25":
        hits = get_bm25().search(req.question, top_k=req.top_k, history=history)
    elif mode == "vector":
        store = get_vector_store()
        bm25 = get_bm25()
        ranked = store.search(req.question, top_k=req.top_k, history=history)
        if ranked and gate_refuses(bm25.tokenized, rewrite_query(req.question, history),
                                   store.index[ranked[0][0]["chunk_id"]], vocab=bm25.idf):
            ranked = []
        hits = ranked
    elif mode == "hybrid":
        hits = get_hybrid().search(req.question, top_k=req.top_k, history=history)
    elif mode == "hybrid_rerank":
        hits = get_hybrid().search_with_rerank(
            req.question, top_k=req.top_k, candidate_k=10, reranker=get_reranker(), history=history)
    else:
        return {"answer": None, "refused": True,
                "reason": f"Unknown retrieval mode: {req.mode!r} (expected bm25, vector, hybrid or hybrid_rerank).",
                "citations": []}
    if not hits:
        return {"answer": None, "refused": True,
                "reason": "No relevant content found in the knowledge base.",
                "citations": []}
    return {"answer": None, "refused": False,
            "note": "LLM generation not wired yet; retrieved passages below.",
            "citations": [{"doc_id": d["doc_id"], "chunk_id": d["chunk_id"],
                           "score": round(s, 3), "text": d["text"][:500]} for d, s in hits]}


@app.post("/ingest")
def ingest():
    global _bm25, _vector_store, _hybrid
    _bm25 = None
    _vector_store = None
    _hybrid = None
    chunks = build_index()
    return {"indexed_chunks": len(chunks)}
