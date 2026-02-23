"""
evaluator.py
------------
Runs all 3 evaluation levels against the RAG pipeline and produces
a detailed report with per-question scores and overall averages.

Usage:
    python evaluate.py
"""

import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.pipeline  import RAGPipeline
from rag.embedder  import Embedder
from ollama        import Client

from evaluation.test_cases import TEST_CASES
from evaluation.metrics    import (
    EvalResult,
    exact_match,
    keyword_match,
    cosine_similarity,
    rouge_scores,
    LLM_JUDGE_PROMPT,
    parse_llm_scores,
)


class RAGEvaluator:
    """
    Evaluates a RAGPipeline across all 3 levels for every test case.

    Usage:
        evaluator = RAGEvaluator(pipeline)
        results   = evaluator.run()
        evaluator.print_report(results)
    """

    def __init__(self, pipeline: RAGPipeline, ollama_model: str = "mistral"):
        self.pipeline  = pipeline
        self.embedder  = Embedder()
        self.client    = Client(host="http://localhost:11434")
        self.llm_model = ollama_model

        # Resolve short model name e.g. 'mistral' → 'mistral:latest'
        available  = [m.model for m in self.client.list().models]
        base_names = [m.split(":")[0] for m in available if m is not None]
        if ollama_model not in available and ollama_model in base_names:
            self.llm_model = available[base_names.index(ollama_model)]

    # ------------------------------------------------------------------
    # Level 3 helper — call LLM judge
    # ------------------------------------------------------------------

    def _llm_judge(
        self,
        question: str,
        expected: str,
        generated: str,
        context_chunks: list[str],
    ) -> tuple[float, float, float, str]:
        """Ask the LLM to score faithfulness, relevance, completeness."""
        context = "\n\n---\n\n".join(context_chunks)
        prompt  = LLM_JUDGE_PROMPT.format(
            question=question,
            context=context,
            expected=expected,
            generated=generated,
        )
        try:
            response = self.client.generate(  # type: ignore[call-overload]
                model=self.llm_model, # pyright: ignore[reportArgumentType]
                prompt=prompt,
                stream=False,
            )
            text = response.response or ""  # type: ignore[union-attr]
            return parse_llm_scores(text)
        except Exception as e:
            return 0.0, 0.0, 0.0, f"LLM judge error: {e}"

    # ------------------------------------------------------------------
    # Main evaluation runner
    # ------------------------------------------------------------------

    def run(self) -> list[EvalResult]:
        """
        Run all test cases through all 3 evaluation levels.

        Returns:
            List of EvalResult objects, one per test case.
        """
        results = []

        print("\n" + "=" * 70)
        print("RUNNING EVALUATION")
        print("=" * 70)

        for i, case in enumerate(TEST_CASES):
            question = case["question"]
            expected = case["expected_answer"]
            category = case["category"]

            print(f"\n[{i + 1}/{len(TEST_CASES)}] {question}")

            # ── Generate answer via pipeline ──────────────────────────
            retrieval = self.pipeline.retrieve(question)
            context_chunks = [chunk for _, _, chunk in retrieval]
            generated = self.pipeline.generator.generate(
                question, context_chunks, stream=False
            )
            print(f"  Generated: {generated[:80]}...")

            result = EvalResult(
                question=question,
                expected_answer=expected,
                generated_answer=generated,
                category=category,
            )

            # ── Level 1: Manual ───────────────────────────────────────
            result.exact_match   = exact_match(generated, expected)
            result.keyword_match = keyword_match(generated, expected)
            print(f"  L1 — Exact: {result.exact_match} | Keywords: {result.keyword_match:.2f}")

            # ── Level 2: Automated ────────────────────────────────────
            gen_emb = self.embedder.encode_query(generated)
            exp_emb = self.embedder.encode_query(expected)
            result.cosine_sim = cosine_similarity(gen_emb, exp_emb)

            rouge = rouge_scores(generated, expected)
            result.rouge1  = rouge["rouge1"]
            result.rouge2  = rouge["rouge2"]
            result.rougeL  = rouge["rougeL"]
            print(f"  L2 — Cosine: {result.cosine_sim:.2f} | R1: {result.rouge1:.2f} | R2: {result.rouge2:.2f} | RL: {result.rougeL:.2f}")

            # ── Level 3: LLM Judge ────────────────────────────────────
            faith, rel, comp, feedback = self._llm_judge(
                question, expected, generated, context_chunks
            )
            result.faithfulness  = faith
            result.relevance     = rel
            result.completeness  = comp
            result.llm_feedback  = feedback
            print(f"  L3 — Faith: {faith:.2f} | Rel: {rel:.2f} | Comp: {comp:.2f}")
            print(f"       Feedback: {feedback}")

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Report printer
    # ------------------------------------------------------------------

    def print_report(self, results: list[EvalResult]) -> None:
        """Print a formatted evaluation report to the terminal."""

        print("\n\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)

        # ── Per-question table ─────────────────────────────────────────
        print(f"\n{'#':<4} {'Category':<18} {'Exact':<7} {'KW':<6} {'Cos':<6} {'R-1':<6} {'R-L':<6} {'Faith':<7} {'Rel':<6} {'Comp':<6}")
        print("-" * 80)

        for i, r in enumerate(results):
            exact = "✓" if r.exact_match else "✗"
            print(
                f"{i+1:<4} {r.category:<18} {exact:<7} "
                f"{r.keyword_match:<6.2f} {r.cosine_sim:<6.2f} "
                f"{r.rouge1:<6.2f} {r.rougeL:<6.2f} "
                f"{r.faithfulness:<7.2f} {r.relevance:<6.2f} {r.completeness:<6.2f}"
            )

        # ── Averages ───────────────────────────────────────────────────
        n = len(results)
        print("\n" + "=" * 70)
        print("OVERALL AVERAGES")
        print("=" * 70)

        exact_pct  = sum(r.exact_match   for r in results) / n * 100
        avg_kw     = sum(r.keyword_match  for r in results) / n
        avg_cos    = sum(r.cosine_sim     for r in results) / n
        avg_r1     = sum(r.rouge1         for r in results) / n
        avg_r2     = sum(r.rouge2         for r in results) / n
        avg_rl     = sum(r.rougeL         for r in results) / n
        avg_faith  = sum(r.faithfulness   for r in results) / n
        avg_rel    = sum(r.relevance      for r in results) / n
        avg_comp   = sum(r.completeness   for r in results) / n

        overall = (avg_cos + avg_rl + avg_faith + avg_rel + avg_comp) / 5

        print(f"\n  Level 1 — Manual")
        print(f"    Exact Match:       {exact_pct:.1f}%  ({sum(r.exact_match for r in results)}/{n} questions)")
        print(f"    Keyword Match:     {avg_kw:.2f} / 1.0")

        print(f"\n  Level 2 — Automated")
        print(f"    Cosine Similarity: {avg_cos:.2f} / 1.0")
        print(f"    ROUGE-1:           {avg_r1:.2f} / 1.0")
        print(f"    ROUGE-2:           {avg_r2:.2f} / 1.0")
        print(f"    ROUGE-L:           {avg_rl:.2f} / 1.0")

        print(f"\n  Level 3 — LLM Judge")
        print(f"    Faithfulness:      {avg_faith:.2f} / 1.0")
        print(f"    Relevance:         {avg_rel:.2f} / 1.0")
        print(f"    Completeness:      {avg_comp:.2f} / 1.0")

        print(f"\n  ★  Overall Pipeline Score: {overall:.2f} / 1.0  ({overall*100:.1f}%)")

        # ── Score interpretation ───────────────────────────────────────
        print("\n" + "=" * 70)
        print("SCORE INTERPRETATION")
        print("=" * 70)
        if overall >= 0.85:
            print("  🟢 EXCELLENT — Pipeline is performing very well")
        elif overall >= 0.70:
            print("  🟡 GOOD — Pipeline is working well with room for improvement")
        elif overall >= 0.50:
            print("  🟠 FAIR — Pipeline needs improvement in some areas")
        else:
            print("  🔴 POOR — Pipeline needs significant improvement")

        # ── Per-category breakdown ─────────────────────────────────────
        print("\n" + "=" * 70)
        print("BREAKDOWN BY CATEGORY")
        print("=" * 70)

        categories: dict[str, list[EvalResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        for cat, cat_results in categories.items():
            cat_score = sum(
                (r.cosine_sim + r.rougeL + r.faithfulness + r.relevance + r.completeness) / 5
                for r in cat_results
            ) / len(cat_results)
            print(f"  {cat:<20} {cat_score:.2f} / 1.0  ({len(cat_results)} questions)")

        print()