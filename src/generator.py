"""
Response generation via Groq API or Ollama.
Produces grounded answers with page citations and fallback when not found.
Uses Groq when GROQ_API_KEY is set; otherwise falls back to Ollama.
"""

from typing import List, Tuple

from .config import (
    FALLBACK_MESSAGE,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)


def format_context(chunks: List[Tuple[str, int, float]]) -> str:
    """
    Format retrieved chunks for the prompt.

    Args:
        chunks: List of (text, page_number, similarity).

    Returns:
        Formatted string like "(Page X) chunk text"
    """
    lines = []
    for text, page, _ in chunks:
        lines.append(f"(Page {page}) {text}")
    return "\n\n".join(lines)


GROUNDED_SYSTEM = """You are a precise assistant. Answer the question using ONLY the provided context. Do NOT use external knowledge. If the answer is not in the context, respond with exactly: "The information is not available in the document." Always cite page numbers as [Page X]."""


def _generate_groq(question: str, context: str, model: str, temperature: float) -> str:
    """Generate response using Groq API."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GROUNDED_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def _generate_ollama(question: str, context: str, model: str, temperature: float) -> str:
    """Generate response using Ollama."""
    from ollama import Client

    prompt = f"""You are a precise assistant. Answer using ONLY the context. If not found, say exactly: "The information is not available in the document." Cite pages as [Page X].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    client = Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
    response = client.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature},
    )
    return (response.get("response", "") or "").strip()


def generate(
    question: str,
    context: str,
    model: str | None = None,
    temperature: float = 0.0,
    fallback: str = FALLBACK_MESSAGE,
) -> str:
    """
    Generate grounded response using Groq (if API key set) or Ollama.

    Args:
        question: User question.
        context: Formatted context from retrieved chunks.
        model: Model name (optional, uses config default).
        temperature: Sampling temperature (0 for deterministic).
        fallback: Message when context is empty.

    Returns:
        Generated answer string.
    """
    if not context.strip():
        return fallback

    if GROQ_API_KEY:
        model = model or GROQ_MODEL
        answer = _generate_groq(question, context, model, temperature)
    else:
        model = model or OLLAMA_MODEL
        answer = _generate_ollama(question, context, model, temperature)

    return answer if answer else fallback
