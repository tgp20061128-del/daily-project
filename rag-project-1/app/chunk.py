"""Markdown chunking: split on headings/paragraphs, ~500 chars, 50 overlap."""
import re

CHUNK_SIZE = 500
OVERLAP = 50


def split_markdown(text):
    """Split text into paragraph blocks, keeping headings attached."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return blocks


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Greedy pack blocks into chunks up to `size` chars, with overlap."""
    blocks = split_markdown(text)
    chunks, current = [], ""
    for block in blocks:
        if len(current) + len(block) + 2 <= size:
            current = f"{current}\n\n{block}".strip() if current else block
        else:
            if current:
                chunks.append(current)
            if len(block) > size:
                # Long block: hard-split with overlap.
                start = 0
                while start < len(block):
                    chunks.append(block[start:start + size])
                    start += size - overlap
                current = ""
            else:
                current = block
    if current:
        chunks.append(current)
    return chunks


def chunk_file(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return chunk_text(text)
