"""Evaluate eval/qa_seed_20.jsonl under three retrieval modes.

Modes (same questions, same hit rule, same refusal gate -- the
comparison measures ranking only):
- bm25:   BM25 (jieba) + min-2-content-overlap refusal gate.
- vector: Chroma vector search + the same gate on the vector top-1 chunk.
- hybrid: BM25 + vector fused with RRF (k=60) + the same gate on the
          fused top-1 chunk.
- hybrid_rerank: hybrid top-10 candidates, reordered by the bge
          cross-encoder to top-3 (gate applied to the fused top-1
          before reranking, so reranking never revives a refusal).
          Only included with --rerank; loads the 1.1 GB model once
          and reuses it for all questions.

Embedding choice: --embedding default (Chroma DefaultEmbeddingFunction,
all-MiniLM-L6-v2) or --embedding bge-small-zh-v1.5 (BAAI Chinese model,
needs sentence-transformers + torch). Output defaults to
eval/results.json; pass --out for a second run (e.g. results_bge.json).

Hit = expected_doc is among the top-k retrieved chunks (k=3).
Refusal items (expected_doc == "") are correct only if nothing is retrieved.
Top-1 accuracy is reported too: with only 4 chunks in the sample corpus
the top-3 rule is coarse (3 of 4 chunks), so top-1 is the number that
actually separates the modes.

Writes results with real, measured numbers. The top level keeps the BM25
run (same shape as before); "comparison" holds all three modes.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.ingest import build_index
from app.retrieval import (
    BM25,
    HybridRetriever,
    VectorStore,
    default_embedding_function,
    gate_refuses,
    get_reranker,
    rewrite_query,
)

BASE = Path(__file__).resolve().parent.parent
QA_PATH = BASE / "eval" / "qa_seed_20.jsonl"
QA_100_PATH = BASE / "eval" / "qa_100.jsonl"
RESULTS_PATH = BASE / "eval" / "results.json"
CHROMA_DIR = BASE / "data" / "chroma"
TOP_K = 3


def _load_items(qa_path=QA_PATH):
    return [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _history_for(item, by_id, rewrite):
    """History texts for one item: follow_up_to chain, oldest first.

    Only multi_turn items carry follow_up_to (in this eval set the chain
    is one level deep, but walk it defensively, cycle-safe, max 3 hops).
    Without --rewrite, or for items without a chain, returns [] -- and
    an empty history leaves retrieval exactly as before.
    """
    if not rewrite or item["category"] != "multi_turn":
        return []
    chain, seen, cur = [], set(), item.get("follow_up_to")
    while cur and cur not in seen and len(chain) < 3:
        seen.add(cur)
        prev = by_id.get(cur)
        if prev is None:
            break
        chain.append(prev["question"])
        cur = prev.get("follow_up_to")
    return list(reversed(chain))


def _run_mode(search_fn, items, by_id, rewrite=False):
    """search_fn(question, history) -> [(doc, score), ...] (top_k)."""
    per_category = defaultdict(lambda: {"total": 0, "correct": 0})
    details = []
    top1_correct = 0
    for item in items:
        history = _history_for(item, by_id, rewrite)
        hits = search_fn(item["question"], history)
        hit_docs = [d["doc_id"] for d, _ in hits]
        if item["expected_doc"]:
            correct = item["expected_doc"] in hit_docs
            top1_ok = bool(hit_docs) and hit_docs[0] == item["expected_doc"]
        else:
            correct = len(hits) == 0
            top1_ok = correct
        top1_correct += int(top1_ok)
        per_category[item["category"]]["total"] += 1
        per_category[item["category"]]["correct"] += int(correct)
        detail = {
            "id": item["id"], "category": item["category"],
            "question": item["question"], "hit_docs": hit_docs,
            "expected_doc": item["expected_doc"], "correct": correct,
        }
        if history:
            detail["history"] = history
            detail["rewritten_question"] = rewrite_query(item["question"], history)
        details.append(detail)
    total = sum(v["total"] for v in per_category.values())
    correct = sum(v["correct"] for v in per_category.values())
    return {
        "total": total, "correct": correct,
        "accuracy": round(correct / total, 3) if total else None,
        "top1_correct": top1_correct,
        "top1_accuracy": round(top1_correct / total, 3) if total else None,
        "per_category": {k: {**v, "accuracy": round(v["correct"] / v["total"], 3)} for k, v in per_category.items()},
        "details": details,
    }


def _dir_size_mb(path):
    """Disk usage in MB, counting each inode once (HF cache hardlinks blobs
    into snapshots/, so a naive sum double-counts)."""
    if not path.exists():
        return None
    seen = set()
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            st = p.stat()
            if st.st_ino not in seen:
                seen.add(st.st_ino)
                total += st.st_size
    return round(total / 1e6, 1)


def _embedding_info(embedding_name):
    if embedding_name == "bge-small-zh-v1.5":
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
        return {
            "model": "BAAI/bge-small-zh-v1.5 (sentence-transformers, 512-dim, normalized)",
            "download_mb": None,  # huggingface_hub snapshot; see cache_size_mb
            "cache_dir": str(cache_dir),
            "cache_size_mb": _dir_size_mb(cache_dir),
            "persist_dir": str(CHROMA_DIR),
        }
    cache_dir = Path.home() / ".cache" / "chroma" / "onnx_models" / VectorStore.EMBEDDING_MODEL
    tarball = cache_dir / "onnx.tar.gz"
    return {
        "model": f"{VectorStore.EMBEDDING_MODEL} (Chroma DefaultEmbeddingFunction, ONNX, 384-dim)",
        "download_mb": round(tarball.stat().st_size / 1e6, 1) if tarball.exists() else None,
        "cache_dir": str(cache_dir),
        "cache_size_mb": _dir_size_mb(cache_dir),
        "persist_dir": str(CHROMA_DIR),
    }


def run(embedding="default", out_path=RESULTS_PATH, qa_path=QA_PATH, rerank=False, rewrite=False):
    items = _load_items(qa_path)
    by_id = {it["id"]: it for it in items}
    n_history = sum(1 for it in items if _history_for(it, by_id, rewrite))
    chunks = build_index()
    n_docs = len({c["doc_id"] for c in chunks})
    bm25 = BM25(chunks)
    vector_store = VectorStore(chunks, CHROMA_DIR, embedding_function=default_embedding_function(embedding))
    hybrid = HybridRetriever(bm25, vector_store, chunks)
    reranker = get_reranker() if rerank else None

    def vector_search(question, history=None):
        ranked = vector_store.search(question, top_k=TOP_K, history=history)
        if not ranked:
            return []
        top_idx = vector_store.index[ranked[0][0]["chunk_id"]]
        if gate_refuses(bm25.tokenized, rewrite_query(question, history), top_idx, vocab=bm25.idf):
            return []
        return ranked

    vec_label = "all-MiniLM-L6-v2" if embedding == "default" else embedding
    rw_suffix = " + query rewrite from follow_up_to history" if rewrite else ""
    comparison = {
        "bm25": {"method": "BM25 (jieba) + min-2-content-overlap refusal gate, top_k=3" + rw_suffix,
                 **_run_mode(lambda q, h: bm25.search(q, top_k=TOP_K, history=h), items, by_id, rewrite)},
        "vector": {"method": f"Chroma vector ({vec_label}) + same refusal gate, top_k=3" + rw_suffix,
                   **_run_mode(vector_search, items, by_id, rewrite)},
        "hybrid": {"method": f"BM25 + vector ({vec_label}), RRF k=60 + same refusal gate, top_k=3" + rw_suffix,
                   **_run_mode(lambda q, h: hybrid.search(q, top_k=TOP_K, history=h), items, by_id, rewrite)},
    }
    if rerank:
        comparison["hybrid_rerank"] = {
            "method": (f"BM25 + vector ({vec_label}), RRF k=60, top-10 candidates reranked "
                       "by BAAI/bge-reranker-base to top_k=3, same refusal gate" + rw_suffix),
            **_run_mode(lambda q, h: hybrid.search_with_rerank(q, top_k=TOP_K, candidate_k=10, reranker=reranker, history=h), items, by_id, rewrite),
        }
    bm25_run = comparison["bm25"]
    results = {
        "method": f"BM25 (jieba) + min-2-content-overlap refusal gate, top_k=3, sample corpus ({n_docs} docs, {len(chunks)} chunks)",
        "qa_file": str(qa_path),
        "corpus": {"docs": n_docs, "chunks": len(chunks)},
        "rewrite": rewrite,
        "history_items": n_history,
        "total": bm25_run["total"], "correct": bm25_run["correct"],
        "accuracy": bm25_run["accuracy"],
        "per_category": bm25_run["per_category"],
        "details": bm25_run["details"],
        "embedding": _embedding_info(embedding),
        "comparison": comparison,
        "note": (f"Corpus is {n_docs} docs / {len(chunks)} chunks; top1_accuracy is the "
                 "more discriminating number when the corpus is small."),
    }
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"qa: {qa_path} ({len(items)} items)  corpus: {n_docs} docs / {len(chunks)} chunks  rewrite: {rewrite} ({n_history} items with history)")
    print(f"embedding: {embedding} -> {out_path}")
    for name, run in comparison.items():
        print(f"{name:>7}: {run['correct']}/{run['total']} acc={run['accuracy']} "
              f"top1={run['top1_correct']}/{run['total']} "
              f"per_cat={ {k: v['correct'] for k, v in run['per_category'].items()} }")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval modes.")
    parser.add_argument("--embedding", default="default", choices=["default", "bge-small-zh-v1.5"])
    parser.add_argument("--qa", default=str(QA_PATH),
                        help="QA jsonl file (default: eval/qa_seed_20.jsonl; 100-item set: eval/qa_100.jsonl)")
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--rerank", action="store_true",
                        help="also run hybrid+rerank (loads BAAI/bge-reranker-base, ~1.1 GB)")
    parser.add_argument("--rewrite", action="store_true",
                        help="expand multi_turn questions with content words from their "
                             "follow_up_to history before retrieval")
    args = parser.parse_args()
    run(embedding=args.embedding, out_path=Path(args.out), qa_path=Path(args.qa), rerank=args.rerank, rewrite=args.rewrite)
