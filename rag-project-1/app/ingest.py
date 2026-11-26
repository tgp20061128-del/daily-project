"""Ingest data/docs/*.md -> chunked index at data/index.json."""
import json
import os
from pathlib import Path

from app.chunk import chunk_text
from app.retrieval import BM25

BASE = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE / "data" / "docs"
INDEX_PATH = BASE / "data" / "index.json"


def build_index(docs_dir=DOCS_DIR):
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({"doc_id": path.name, "chunk_id": f"{path.name}#{i}", "text": chunk})
    return chunks


def main():
    chunks = build_index()
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"indexed {len(chunks)} chunks from {DOCS_DIR} -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
