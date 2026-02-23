"""
generator.py
------------
Handles prompt construction and answer generation via a local Ollama model.
"""

from ollama import Client, GenerateResponse

DEFAULT_HOST  = "http://localhost:11434"
DEFAULT_MODEL = "mistral"


class Generator:
    """
    Connects to a local Ollama instance and generates answers using RAG context.

    Usage:
        gen = Generator()
        answer = gen.generate(question, context_chunks, stream=True)
    """

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL):
        """
        Connect to Ollama and verify the connection.

        Args:
            host:  Ollama server URL.
            model: Name of the Ollama model to use (e.g. 'mistral', 'llama3').
        """
        self.model  = model
        self.client = Client(host=host)

        available  = [m.model for m in self.client.list().models]

        # Fix: m.model can be None — guard before calling split()
        base_names = [m.split(":")[0] for m in available if m is not None]

        print(f"✓ Connected to Ollama  |  available models: {available}")

        if model not in available and model not in base_names:
            # Model not found at all
            print(f"  ⚠  '{model}' not found locally. Run: ollama pull {model}")
        elif model not in available and model in base_names:
            # Resolve short name → full tag  e.g. 'mistral' → 'mistral:latest'
            self.model = available[base_names.index(model)]
            print(f"  ✓ Resolved '{model}' → '{self.model}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, context: str) -> str:
        """Assemble the RAG prompt from question and retrieved context."""
        return (
            "You are a helpful assistant. Answer the following question based "
            "ONLY on the provided context. If the answer is not in the context, "
            "say 'I could not find this information in the document.'\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

    @staticmethod
    def _extract_token(chunk: GenerateResponse) -> str:
        """
        Extract the text token from an Ollama GenerateResponse object.
        """
        return chunk.response or ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        chunks: list[str],
        stream: bool = True,
    ) -> str:
        """
        Generate an answer for *question* grounded in *chunks*.

        Args:
            question: The user's question.
            chunks:   Retrieved text chunks used as context.
            stream:   Whether to stream tokens to stdout as they arrive.

        Returns:
            The full generated answer as a string.
        """
        context = "\n\n---\n\n".join(chunks)
        prompt  = self._build_prompt(question, context)

        full_answer = ""

        if stream:
            print("ANSWER:")
            print("=" * 70)
            # Fix: pass stream=True explicitly via the correct overload signature
            for chunk in self.client.generate(model=self.model, prompt=prompt, stream=True):  # type: ignore[call-overload]
                token = self._extract_token(chunk)
                print(token, end="", flush=True)
                full_answer += token
            print(f"\n{'=' * 70}\n")
        else:
            response    = self.client.generate(model=self.model, prompt=prompt, stream=False) # pyright: ignore[reportArgumentType, reportCallIssue]
            full_answer = self._extract_token(response)  # type: ignore[arg-type]

        # Guard against empty responses
        if not full_answer.strip():
            full_answer = "I could not find this information in the document."
            print(full_answer)

        return full_answer