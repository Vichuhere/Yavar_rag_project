"""
Embedding generation using sentence-transformers (all-MiniLM-L6-v2).
Supports batch encoding for efficiency.
"""

from typing import List

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class Embedder:
    """Loads sentence-transformers model and encodes texts to embeddings."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
