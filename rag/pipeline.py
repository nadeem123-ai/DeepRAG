"""
pipeline.py
-----------
Orchestrates the full RAG pipeline:
    PDF → Text → Chunks → Embeddings → FAISS Index → Retrieve → Generate

All expensive steps (extraction, embedding) are cached to disk so they only
run once. Delete the cache files to force a full rebuild.
"""

import os
import pickle

from rag.pdf_extractor import extract_text
from rag.chunker      import chunk_text
from rag.embedder     import Embedder
from rag.vector_store import VectorStore
from rag.generator    import Generator

CACHE_FILE = "rag_cache.pkl"


class RAGPipeline:
    """
    End-to-end RAG pipeline with automatic caching.

    Usage:
        pipeline = RAGPipeline(pdf_path="resume.pdf")
        answer   = pipeline.ask("What are this person's skills?")
    """

    def __init__(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        overlap: int = 150,
        top_k: int = 5,
        index_type: str = "flat",
        ollama_model: str = "mistral",
        cache_file: str = CACHE_FILE,
    ):
        self.top_k      = top_k
        self.cache_file = cache_file

        print("=" * 70)
        print("Initialising RAG Pipeline")
        print("=" * 70)

        # ── 1. Load or build chunks + embeddings ───────────────────────
        chunks, embeddings = self._load_or_build(pdf_path, chunk_size, overlap)

        # ── 2. Build vector store ──────────────────────────────────────
        self.store = VectorStore(embeddings, chunks, index_type=index_type)

        # ── 3. Keep embedder alive for query encoding ──────────────────
        #    (model already loaded inside _load_or_build; reuse it)
        self.embedder = self._embedder

        # ── 4. Connect generator ───────────────────────────────────────
        self.generator = Generator(model=ollama_model)

        print("\n✓ Pipeline ready\n")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_or_build(self, pdf_path: str, chunk_size: int, overlap: int):
        """Return (chunks, embeddings), using cache when available."""

        if os.path.exists(self.cache_file):
            print(f"[Cache] Loading from '{self.cache_file}'...")
            with open(self.cache_file, "rb") as f:
                cache = pickle.load(f)
            chunks     = cache["chunks"]
            embeddings = cache["embeddings"]
            print(f"✓ Loaded {len(chunks)} chunks from cache")

            # We still need an Embedder to encode queries at search time
            self._embedder = Embedder()
        else:
            print("[Cache] No cache found — building from scratch...")

            # Step 1: Extract
            print("\n[1/3] Extracting text from PDF...")
            text = extract_text(pdf_path)
            print(f"✓ Extracted {len(text):,} characters")

            # Step 2: Chunk
            print("\n[2/3] Chunking text...")
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            print(f"✓ Created {len(chunks)} chunks")

            # Step 3: Embed
            print("\n[3/3] Creating embeddings...")
            self._embedder = Embedder()
            embeddings     = self._embedder.encode(chunks)
            print(f"✓ Embeddings shape: {embeddings.shape}")

            # Save cache
            with open(self.cache_file, "wb") as f:
                pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
            print(f"\n✓ Cache saved to '{self.cache_file}'")

        return chunks, embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[tuple]:
        """
        Retrieve the most relevant chunks for *question*.

        Returns:
            List of (similarity, chunk_index, chunk_text) tuples.
        """
        query_vec = self.embedder.encode_query(question)
        return self.store.search(query_vec, k=self.top_k)

    def ask(self, question: str, stream: bool = True) -> str:
        """
        Answer *question* using the full RAG pipeline.

        Args:
            question: The user's question.
            stream:   Stream tokens to stdout while generating.

        Returns:
            The generated answer string.
        """
        print(f"\n{'=' * 70}")
        print(f"Question: {question}")
        print("=" * 70)

        # Retrieve
        results = self.retrieve(question)
        print(f"\nRetrieved {len(results)} chunks:")
        for i, (sim, idx, _) in enumerate(results):
            print(f"  {i + 1}. Chunk {idx + 1}  (relevance: {sim:.3f})")

        # Generate
        context_chunks = [chunk for _, _, chunk in results]
        return self.generator.generate(question, context_chunks, stream=stream)