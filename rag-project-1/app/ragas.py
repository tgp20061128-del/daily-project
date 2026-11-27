"""RAGAS-style retrieval metrics, computed from ground truth only.

No LLM is involved anywhere here: relevance is decided by the eval
set's ``expected_doc`` label, not by an LLM judge. That is exactly the
part of RAGAS that can be done without an API key -- and the part that
cannot (faithfulness, answer relevancy) is out of scope until /ask
generates answers; see README "RAGAS 检索侧评测".

Ground truth
------------
The eval set labels each question with ONE document (``expected_doc``);
there is no chunk-level label and no multi-chunk label. So the relevant
set for a question is defined as **all chunks of its expected doc**:

    relevant(chunk)  <=>  chunk["doc_id"] == expected_doc

Two consequences, stated plainly so the numbers are not misread:

1. Context recall is chunk coverage of the whole source doc, not
   "was the answer-bearing chunk found". With top_k=3 and a doc of N
   chunks, the best achievable recall is min(3, N) / N -- 1.0 for the
   two 2-chunk docs, only 3/16 = 0.1875 for the 16-chunk doc. The
   report therefore also gives ``max_recall_at_k`` (the mean of that
   ceiling over the scored items) so a recall of ~0.25 can be read
   against its ceiling instead of against 1.0.
2. The answer to a question usually lives in one chunk, so a recall
   below 1.0 does NOT mean the answer was missed -- Hit@3 / MRR are
   the numbers that say whether the right doc was found at all.

Context Precision follows the RAGAS definition: rank-weighted
precision over the retrieved list,

    CP = sum_{k=1..K} P@k * v_k / sum_{k=1..K} v_k

where v_k = 1 iff the k-th retrieved chunk is relevant and P@k is
precision among the first k retrieved chunks. The denominator is the
number of relevant chunks *in the retrieved list* (not in the whole
corpus), exactly as RAGAS defines it. With no relevant chunk
retrieved, CP = 0. This rewards putting relevant chunks early: a
relevant chunk at rank 1 counts more than the same chunk at rank 3.

Refusal items (expected_doc == "") are NOT scored here: refusing is a
gate decision, not a retrieval result. They are counted separately
(refusal correct / total) by the evaluation driver.
"""

TOP_K = 3


def _is_relevant(doc_id, expected_doc):
    return bool(expected_doc) and doc_id == expected_doc


def context_precision(retrieved_doc_ids, expected_doc):
    """RAGAS Context Precision over a ranked list of doc ids.

    retrieved_doc_ids: doc ids in rank order (best first).
    Returns a float in [0, 1]; 0 when no retrieved chunk is relevant.
    """
    if not expected_doc or not retrieved_doc_ids:
        return 0.0
    relevant_so_far = 0
    weighted_sum = 0.0
    for k, doc_id in enumerate(retrieved_doc_ids, start=1):
        if _is_relevant(doc_id, expected_doc):
            relevant_so_far += 1
            weighted_sum += relevant_so_far / k
    if relevant_so_far == 0:
        return 0.0
    return weighted_sum / relevant_so_far


def context_recall(retrieved_doc_ids, expected_doc, n_chunks_in_doc):
    """RAGAS Context Recall: relevant chunks retrieved / chunks in doc.

    n_chunks_in_doc is the total chunk count of the expected doc.
    Returns 0.0 when nothing relevant was retrieved; None when the
    item has no ground-truth doc (refusal item -- caller must exclude
    those instead of scoring them).
    """
    if not expected_doc:
        return None
    if n_chunks_in_doc <= 0:
        return 0.0
    retrieved_relevant = sum(1 for d in retrieved_doc_ids if _is_relevant(d, expected_doc))
    return retrieved_relevant / n_chunks_in_doc


def max_recall_at_k(n_chunks_in_doc, k=TOP_K):
    """Ceiling of context_recall at top-k: min(k, N) / N."""
    if n_chunks_in_doc <= 0:
        return 0.0
    return min(k, n_chunks_in_doc) / n_chunks_in_doc


def reciprocal_rank(retrieved_doc_ids, expected_doc):
    """MRR contribution: 1 / rank of the first relevant chunk, else 0."""
    if not expected_doc:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if _is_relevant(doc_id, expected_doc):
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_doc_ids, expected_doc, k):
    """True when a relevant chunk appears within the first k results."""
    if not expected_doc:
        return False
    return any(_is_relevant(d, expected_doc) for d in retrieved_doc_ids[:k])
