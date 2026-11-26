# daily-project

Finished projects, one folder per project. Updated whenever a project is done.

## Projects

| Project | What it is | Status |
|---|---|---|
| [rag-project-1](rag-project-1/) | Enterprise RAG knowledge base with evaluation: BM25 + Chroma vector + RRF hybrid + bge-reranker + multi-turn query rewriting + refusal gate | Done (retrieval pipeline) |

## rag-project-1 results (measured locally, 100-item eval set, 7 docs / 76 chunks)

| Mode | Top-3 hit | Top-1 |
|---|---|---|
| BM25 + query rewriting | 100/100 | 88/100 |
| Hybrid (BM25 + vector, RRF) + rewriting | 97/100 | 90/100 |
| Hybrid + reranker + rewriting | 97/100 | 92/100 |

Refusal (no-answer) questions: 20/20. See [rag-project-1/README.md](rag-project-1/README.md) for method, limitations, and how to run it.

## How this repo is maintained

Each finished project lands here as its own folder with its own README, code, eval set, and measured results. No generated artifacts (virtualenvs, vector-store data, caches) are committed.
