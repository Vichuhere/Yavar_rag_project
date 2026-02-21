"""
Text chunking with configurable size and overlap.
Preserves page_number in metadata for source citations.
"""

from typing import Iterator

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    pages: Iterator[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Iterator[tuple[str, dict]]:
    """
    Split page texts into overlapping chunks with metadata.

    Args:
        pages: Iterator of (page_number, page_text).
        chunk_size: Target size of each chunk in characters.
        overlap: Number of characters to overlap between consecutive chunks.

    Yields:
        Tuples of (chunk_text, metadata) where metadata contains
        page_number and chunk_id.
    """
    chunk_id = 0
    step = chunk_size - overlap

    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunk_id += 1
                metadata = {"page_number": page_num, "chunk_id": chunk_id}
                yield chunk, metadata
            start += step
