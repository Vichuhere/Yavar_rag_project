"""
Retrieval logic with similarity threshold filtering.
Returns empty result if no chunk passes the threshold.
"""

from typing import List, Optional, Tuple

from .config import SIMILARITY_THRESHOLD, TOP_K
from .embedder import Embedder
from .vector_db import VectorDB


def _cosine_to_similarity(distance: float) -> float:
    """
    ChromaDB with cosine distance returns values in [0, 2].
    Convert to similarity: 1 - (distance / 2) gives [1, 0] for [0, 2].
    Alternatively, similarity = 1 - distance for cosine distance.
    """
    # ChromaDB cosine "distance" is 1 - cosine_sim, so similarity = 1 - distance
    return 1.0 - distance


class Retriever:
    """
    Retrieves relevant chunks using embeddings and applies
    a similarity threshold to avoid low-quality matches.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        embedder: Embedder,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.vector_db = vector_db
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Tuple[str, int, float]]:
        """
        Retrieve top-k chunks that exceed the similarity threshold.

        Args:
            query: User question.

        Returns:
            List of (chunk_text, page_number, similarity_score).
            Empty if no chunks pass the threshold or DB is empty.
        """
        if self.vector_db._collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query])[0]
        result = self.vector_db.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        if not ids:
            return []

        results = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            sim = _cosine_to_similarity(float(dist))
            if sim >= self.similarity_threshold:
                page = meta.get("page_number", 0)
                results.append((doc, page, sim))

        return results
