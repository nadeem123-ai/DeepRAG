"""
embedder.py
-----------
Converts text chunks into dense vector embeddings using SentenceTransformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Default model — change here to swap globally across the whole project
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps a SentenceTransformer model to encode text into embeddings.

    Usage:
        embedder = Embedder()
        embeddings = embedder.encode(chunks)        # encode a list of chunks
        query_vec  = embedder.encode_query("hello") # encode a single query
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        print("✓ Embedding model loaded")

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into a 2-D numpy array of embeddings.

        Args:
            texts:         List of strings to encode.
            show_progress: Whether to display a progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        return self.model.encode(texts, show_progress_bar=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a 1-D embedding vector.

        Args:
            query: The user's question or search string.

        Returns:
            1-D numpy array of shape (embedding_dim,).
        """
        return self.model.encode(query) # pyright: ignore[reportReturnType]

