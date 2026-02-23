"""
chunker.py
----------
Splits extracted text into overlapping chunks suitable for embedding.
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Break text into overlapping chunks to preserve context at boundaries.

    Args:
        text:       Full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks
