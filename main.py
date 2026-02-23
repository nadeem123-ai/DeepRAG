"""
main.py
-------
Entry point for the RAG system.
Runs a short demo, then drops into an interactive Q&A loop.

Usage:
    python main.py
    python main.py --pdf /path/to/your.pdf --model mistral
"""

import argparse

from rag import RAGPipeline

# ── Demo questions shown on startup ───────────────────────────────────────────
DEMO_QUESTIONS = [
    "Who is Muhammad Nadeem?",
    "What are his technical skills?"
]


def run_demo(pipeline: RAGPipeline) -> None:
    """Run a short demo with preset questions."""
    print("\n" + "=" * 70)
    print("DEMO MODE")
    print("=" * 70)

    for question in DEMO_QUESTIONS:
        pipeline.ask(question)
        input("\nPress Enter for the next question...")


def run_interactive(pipeline: RAGPipeline) -> None:
    """Drop into an interactive Q&A loop."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE  —  type 'exit' to quit")
    print("=" * 70 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            pipeline.ask(question)
        except Exception as e:
            print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Q&A over a PDF document")
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
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per query (default: 3)",
    )
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip the demo questions and go straight to interactive mode",
    )
    args = parser.parse_args()

    # Build the pipeline once — all steps are cached after the first run
    pipeline = RAGPipeline(
        pdf_path=args.pdf,
        top_k=args.top_k,
        ollama_model=args.model,
    )

    if not args.no_demo:
        run_demo(pipeline)

    run_interactive(pipeline)


if __name__ == "__main__":
    main()