"""
vector_store.py
---------------
Manages a FAISS index for fast similarity search over chunk embeddings.
"""

import numpy as np
import faiss


class VectorStore:
    """
    Stores chunk embeddings in a FAISS index and provides similarity search.

    Two index types are supported:
      - 'flat' : Exact exhaustive search. Always accurate. Best for small datasets.
      - 'ivf'  : Approximate search using clustering. Faster for large datasets.

    Usage:
        store = VectorStore(embeddings, chunks)
        results = store.search(query_embedding, k=3)
    """

    def __init__(self, embeddings: np.ndarray, chunks: list[str], index_type: str = "flat"):
        """
        Build the FAISS index from the provided embeddings.

        Args:
            embeddings: numpy array of shape (num_chunks, embedding_dim).
            chunks:     Corresponding list of text chunks.
            index_type: 'flat' (default) or 'ivf'.
        """
        self.chunks = chunks
        self.embedding_dim = embeddings.shape[1]

        embeddings_fp32 = embeddings.astype("float32")

        if index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings_fp32)

        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            nlist = max(1, min(100, len(embeddings) // 8))
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings_fp32)
            self.index.add(embeddings_fp32)

        else:
            raise ValueError(f"Unknown index_type '{index_type}'. Use 'flat' or 'ivf'.")

        print(f"✓ FAISS '{index_type}' index built with {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[tuple]:
        """
        Find the k most similar chunks to the query embedding.

        Args:
            query_embedding: 1-D numpy array of shape (embedding_dim,).
            k:               Number of results to return.

        Returns:
            List of (similarity_score, chunk_index, chunk_text) tuples,
            sorted by descending similarity.
        """
        query_2d = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_2d, k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + distance)   # convert L2 distance → similarity score
            results.append((similarity, int(idx), self.chunks[idx]))

        return results
