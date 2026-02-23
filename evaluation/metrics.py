"""
metrics.py
----------
All evaluation metrics for the RAG pipeline across 3 levels.

Level 1 — Manual:      exact match, keyword match
Level 2 — Automated:   cosine similarity, ROUGE-1, ROUGE-2, ROUGE-L
Level 3 — LLM Judge:   faithfulness, relevance, completeness scored by Ollama
"""

import re
import numpy as np
from dataclasses import dataclass, field


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RetrievalResult:
    """Holds retrieval output for one question."""
    question:    str
    chunks:      list[str]
    similarities: list[float]


@dataclass
class EvalResult:
    """Holds all evaluation scores for one test case."""
    question:         str
    expected_answer:  str
    generated_answer: str
    category:         str

    # Level 1
    exact_match:      bool  = False
    keyword_match:    float = 0.0

    # Level 2
    cosine_sim:       float = 0.0
    rouge1:           float = 0.0
    rouge2:           float = 0.0
    rougeL:           float = 0.0

    # Level 3
    faithfulness:     float = 0.0
    relevance:        float = 0.0
    completeness:     float = 0.0
    llm_feedback:     str   = ""


# ============================================================================
# Level 1 — Manual Metrics
# ============================================================================

def exact_match(generated: str, expected: str) -> bool:
    """
    Check if the expected answer appears anywhere in the generated answer.
    Case-insensitive.
    """
    return expected.lower().strip() in generated.lower().strip()


def keyword_match(generated: str, expected: str) -> float:
    """
    What fraction of keywords in the expected answer appear in the generated answer.
    Ignores stopwords.

    Returns a score between 0.0 and 1.0.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "of", "and", "or", "for", "with", "his", "her", "their",
        "he", "she", "it", "this", "that", "has", "have", "had", "be",
        "been", "being", "as", "by", "from", "what", "which", "who"
    }

    def extract_keywords(text: str) -> set[str]:
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    expected_kws  = extract_keywords(expected)
    generated_kws = extract_keywords(generated)

    if not expected_kws:
        return 0.0

    matched = expected_kws & generated_kws
    return len(matched) / len(expected_kws)


# ============================================================================
# Level 2 — Automated Metrics
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Returns a score between -1.0 and 1.0 (higher = more similar).
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    """Helper to extract n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_scores(generated: str, expected: str) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    ROUGE measures word overlap between generated and expected answers:
    - ROUGE-1: unigram (single word) overlap
    - ROUGE-2: bigram (two word) overlap
    - ROUGE-L: longest common subsequence

    Returns dict with keys: rouge1, rouge2, rougeL  (all F1 scores 0.0–1.0)
    """
    gen_tokens = re.findall(r'\b\w+\b', generated.lower())
    exp_tokens = re.findall(r'\b\w+\b', expected.lower())

    def f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def ngram_f1(n: int) -> float:
        gen_ng = _get_ngrams(gen_tokens, n)
        exp_ng = _get_ngrams(exp_tokens, n)
        if not exp_ng or not gen_ng:
            return 0.0
        gen_set = {}
        for ng in gen_ng:
            gen_set[ng] = gen_set.get(ng, 0) + 1
        exp_set = {}
        for ng in exp_ng:
            exp_set[ng] = exp_set.get(ng, 0) + 1
        overlap = sum(min(gen_set.get(ng, 0), cnt) for ng, cnt in exp_set.items())
        precision = overlap / len(gen_ng)
        recall    = overlap / len(exp_ng)
        return f1(precision, recall)

    def lcs_length(a: list, b: list) -> int:
        """Compute length of longest common subsequence."""
        m, n  = len(a), len(b)
        dp    = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def rouge_l() -> float:
        if not exp_tokens or not gen_tokens:
            return 0.0
        lcs       = lcs_length(gen_tokens, exp_tokens)
        precision = lcs / len(gen_tokens)
        recall    = lcs / len(exp_tokens)
        return f1(precision, recall)

    return {
        "rouge1": ngram_f1(1),
        "rouge2": ngram_f1(2),
        "rougeL": rouge_l(),
    }


# ============================================================================
# Level 3 — LLM as Judge
# ============================================================================

LLM_JUDGE_PROMPT = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the generated answer against the question and context provided.
Score each dimension from 0.0 to 1.0 (use decimals like 0.7, 0.85 etc).

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

EXPECTED ANSWER: {expected}

GENERATED ANSWER: {generated}

Score these 3 dimensions:

1. FAITHFULNESS (0.0-1.0): Is the generated answer grounded in the retrieved context?
   - 1.0 = every claim is supported by the context
   - 0.5 = some claims are supported, some are not
   - 0.0 = answer contradicts or ignores the context

2. RELEVANCE (0.0-1.0): Does the generated answer actually address the question?
   - 1.0 = directly and completely answers the question
   - 0.5 = partially answers the question
   - 0.0 = does not answer the question at all

3. COMPLETENESS (0.0-1.0): Does the generated answer cover everything in the expected answer?
   - 1.0 = covers all key points from expected answer
   - 0.5 = covers some key points
   - 0.0 = misses all key points

Respond ONLY in this exact format (no extra text):
FAITHFULNESS: <score>
RELEVANCE: <score>
COMPLETENESS: <score>
FEEDBACK: <one sentence explaining the scores>
"""


def parse_llm_scores(response: str) -> tuple[float, float, float, str]:
    """
    Parse the structured LLM judge response into scores.

    Returns:
        (faithfulness, relevance, completeness, feedback)
    """
    faithfulness  = 0.0
    relevance     = 0.0
    completeness  = 0.0
    feedback      = "Could not parse LLM response."

    try:
        for line in response.strip().splitlines():
            line = line.strip()
            if line.startswith("FAITHFULNESS:"):
                faithfulness = float(line.split(":")[1].strip())
            elif line.startswith("RELEVANCE:"):
                relevance = float(line.split(":")[1].strip())
            elif line.startswith("COMPLETENESS:"):
                completeness = float(line.split(":")[1].strip())
            elif line.startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()
    except (ValueError, IndexError):
        pass

    return faithfulness, relevance, completeness, feedback