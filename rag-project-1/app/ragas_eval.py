"""RAGAS-style retrieval evaluation (ground truth only, no LLM).

Runs every retrieval mode on eval/qa_100.jsonl and scores each question
with the pure functions in app/ragas.py:

- Context Precision (RAGAS rank-weighted formula)
- Context Recall (retrieved relevant chunks / chunks in the expected doc)
- MRR, Hit@1, Hit@3 (doc-level: a chunk counts when its doc is the
  expected doc)

Refusal items (the 20 no_answer questions, expected_doc == "") are
excluded from every retrieval-metric denominator -- refusing is a gate
decision, not a retrieval result -- and are counted separately as
refusal_correct / refusal_total per mode.

Usage:
    # smoke test, 5 items, BM25 only
    .venv/bin/python -m app.ragas_eval --limit 5 --modes bm25
    # full run, all non-rerank modes x rewrite on/off (bge embeddings)
    .venv/bin/python -m app.ragas_eval --out eval/results_ragas.json
    # rerank modes only (slow: ~20 min CPU for 100 items); merges into
    # the same output file, keeping the earlier modes
    .venv/bin/python -m app.ragas_eval --rerank --modes hybrid_rerank \\
        --out eval/results_ragas.json

Embedding default is bge-small-zh-v1.5, matching the earlier 100-item
result files (results_100*.json). Rerank mode also needs the
bge-reranker-base cross-encoder (~1.1 GB, cached).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evaluate import _history_for, _load_items
from app.ingest import build_index
from app.ragas import (
    TOP_K,
    context_precision,
    context_recall,
    hit_at_k,
    max_recall_at_k,
    reciprocal_rank,
)
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
QA_PATH = BASE / "eval" / "qa_100.jsonl"
RESULTS_PATH = BASE / "eval" / "results_ragas.json"
CHROMA_DIR = BASE / "data" / "chroma"

ALL_MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]


def _score_items(search_fn, items, by_id, chunks_per_doc, rewrite=False):
    """Score one mode: RAGAS metrics over non-refusal items + refusal count.

    search_fn(question, history) -> [(doc, score), ...] (top_k), already
    gated exactly like /ask (empty list = refused).
    """
    sums = defaultdict(float)
    per_category = defaultdict(lambda: defaultdict(float))
    per_category_n = Counter()
    n_scored = 0
    refusal_correct = 0
    refusal_total = 0
    details = []
    for item in items:
        history = _history_for(item, by_id, rewrite)
        hits = search_fn(item["question"], history)
        hit_docs = [d["doc_id"] for d, _ in hits]
        if not item["expected_doc"]:
            refusal_total += 1
            refusal_correct += int(len(hits) == 0)
            continue
        n_scored += 1
        n_doc = chunks_per_doc[item["expected_doc"]]
        cp = context_precision(hit_docs, item["expected_doc"])
        cr = context_recall(hit_docs, item["expected_doc"], n_doc)
        mrr = reciprocal_rank(hit_docs, item["expected_doc"])
        h1 = float(hit_at_k(hit_docs, item["expected_doc"], 1))
        h3 = float(hit_at_k(hit_docs, item["expected_doc"], 3))
        ceil = max_recall_at_k(n_doc)
        for key, val in [("context_precision", cp), ("context_recall", cr),
                         ("mrr", mrr), ("hit_at_1", h1), ("hit_at_3", h3),
                         ("max_recall_at_k", ceil)]:
            sums[key] += val
            per_category[item["category"]][key] += val
        per_category_n[item["category"]] += 1
        details.append({
            "id": item["id"], "category": item["category"],
            "expected_doc": item["expected_doc"],
            "hit_docs": hit_docs,
            "context_precision": round(cp, 4),
            "context_recall": round(cr, 4),
            "mrr": round(mrr, 4),
            "hit_at_1": int(h1), "hit_at_3": int(h3),
        })
    metrics = {k: round(v / n_scored, 4) if n_scored else None for k, v in sums.items()}
    per_cat = {
        cat: {"n": per_category_n[cat],
              **{k: round(v / per_category_n[cat], 4) for k, v in vals.items()}}
        for cat, vals in sorted(per_category.items())
    }
    return {
        "rewrite": rewrite,
        "n_scored": n_scored,
        "refusal_correct": refusal_correct,
        "refusal_total": refusal_total,
        **metrics,
        "per_category": per_cat,
        "details": details,
    }


def run(embedding="bge-small-zh-v1.5", out_path=RESULTS_PATH, qa_path=QA_PATH,
        modes=None, rerank=False, rewrite_both=True, limit=None):
    modes = list(modes) if modes else ["bm25", "vector", "hybrid"]
    if rerank and "hybrid_rerank" not in modes:
        modes.append("hybrid_rerank")
    items = _load_items(qa_path)
    if limit:
        items = items[:limit]
    by_id = {it["id"]: it for it in items}
    chunks = build_index()
    chunks_per_doc = dict(Counter(c["doc_id"] for c in chunks))
    n_docs = len(chunks_per_doc)
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

    search_fns = {
        "bm25": lambda q, h: bm25.search(q, top_k=TOP_K, history=h),
        "vector": vector_search,
        "hybrid": lambda q, h: hybrid.search(q, top_k=TOP_K, history=h),
        "hybrid_rerank": lambda q, h: hybrid.search_with_rerank(
            q, top_k=TOP_K, candidate_k=10, reranker=reranker, history=h),
    }

    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        results = {
            "method": ("RAGAS-style retrieval metrics at top_k=3, ground truth only, "
                       "no LLM. Relevance: chunk is relevant iff its doc_id equals "
                       "the item's expected_doc (doc-level labels; no chunk-level "
                       "labels exist in the eval set)."),
            "qa_file": str(qa_path),
            "corpus": {"docs": n_docs, "chunks": len(chunks),
                       "chunks_per_doc": chunks_per_doc},
            "top_k": TOP_K,
            "embedding": embedding,
            "sample": {"total": len(_load_items(qa_path)),
                       "scored_per_run": sum(1 for it in _load_items(qa_path) if it["expected_doc"]),
                       "refusal_excluded_per_run": sum(1 for it in _load_items(qa_path) if not it["expected_doc"])},
            "definitions": {
                "context_precision": "RAGAS: sum_k P@k * v_k / sum_k v_k over the top-3 ranked chunks; v_k=1 iff chunk k is relevant. Rewards relevant chunks ranked early.",
                "context_recall": "Relevant chunks retrieved in top-3 / total chunks in the expected doc. Ceiling at top-3 is min(3, N)/N (see max_recall_at_k).",
                "max_recall_at_k": "Mean over scored items of min(3, N)/N -- the best recall achievable at top-3 given doc sizes.",
                "mrr": "Mean reciprocal rank of the first relevant chunk (doc-level), 0 if none in top-3.",
                "hit_at_1": "Share of scored items whose top-1 chunk is from the expected doc.",
                "hit_at_3": "Share of scored items whose expected doc appears anywhere in top-3.",
                "refusal_handling": "The 20 no_answer items are excluded from every retrieval-metric denominator (n_scored=80 per full run); refusal is a gate decision, counted separately as refusal_correct/refusal_total per mode.",
            },
            "modes": {},
        }
    rewrite_settings = [False, True] if rewrite_both else [False]
    for mode in modes:
        for rw in rewrite_settings:
            key = mode if not rw else f"{mode}+rewrite"
            print(f"running {key} ...", flush=True)
            results["modes"][key] = _score_items(
                search_fns[mode], items, by_id, chunks_per_doc, rewrite=rw)
            r = results["modes"][key]
            print(f"  {key}: n={r['n_scored']} CP={r['context_precision']} "
                  f"CR={r['context_recall']} (max {r['max_recall_at_k']}) "
                  f"MRR={r['mrr']} H@1={r['hit_at_1']} H@3={r['hit_at_3']} "
                  f"refusal={r['refusal_correct']}/{r['refusal_total']}", flush=True)
            # Write after every mode, not just at the end: a rerank run
            # takes ~20 min per setting on CPU, and an interrupted run
            # must not lose the modes that already finished.
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS-style retrieval evaluation (no LLM).")
    parser.add_argument("--embedding", default="bge-small-zh-v1.5",
                        choices=["default", "bge-small-zh-v1.5"])
    parser.add_argument("--qa", default=str(QA_PATH))
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--modes", default="bm25,vector,hybrid",
                        help="comma-separated subset of bm25,vector,hybrid,hybrid_rerank")
    parser.add_argument("--rerank", action="store_true",
                        help="include hybrid_rerank (loads bge-reranker-base, slow on CPU)")
    parser.add_argument("--no-rewrite", dest="rewrite_both", action="store_false",
                        help="run without query rewrite only (skip the +rewrite runs)")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N items (smoke test)")
    args = parser.parse_args()
    run(embedding=args.embedding, out_path=Path(args.out), qa_path=Path(args.qa),
        modes=[m.strip() for m in args.modes.split(",") if m.strip()],
        rerank=args.rerank, rewrite_both=args.rewrite_both, limit=args.limit)
