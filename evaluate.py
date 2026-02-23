"""
evaluate.py
-----------
Entry point for running the full RAG pipeline evaluation.

Usage:
    python evaluate.py
    python evaluate.py --pdf docs/Nadeem_Updated_Resume_26.pdf --model mistral
"""

import argparse
import os
from rag        import RAGPipeline
from evaluation import RAGEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument(
        "--pdf",
        default="docs/Nadeem_Updated_Resume_26.pdf",
        help="Path to the source PDF file",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Ollama model name (default: mistral)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Delete existing cache and rebuild from scratch",
    )
    args = parser.parse_args()

    # Delete cache if requested — forces rebuild with new chunk settings
    if args.rebuild_cache and os.path.exists("rag_cache.pkl"):
        os.remove("rag_cache.pkl")
        print("✓ Cache deleted — will rebuild with new settings\n")

    # Build pipeline (uses cache if available)
    pipeline  = RAGPipeline(pdf_path=args.pdf, ollama_model=args.model)

    # Run evaluation
    evaluator = RAGEvaluator(pipeline, ollama_model=args.model)
    results   = evaluator.run()

    # Print full report
    evaluator.print_report(results)


if __name__ == "__main__":
    main()