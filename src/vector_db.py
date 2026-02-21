"""
Vector database using ChromaDB in persistent local mode.
Stores document text, embeddings, and metadata (page_number, chunk_id).
"""

from typing import List, Optional

import chromadb
from chromadb.config import Settings

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
)


class VectorDB:
    """
    ChromaDB-backed vector store with persistent storage.
    Clears existing collection before indexing new document.
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        """Remove all documents from the collection (for re-indexing)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        """
        Add documents with embeddings and metadata to the collection.

        Args:
            ids: Unique identifiers for each chunk.
            documents: Raw text of each chunk.
            embeddings: Embedding vectors.
            metadatas: Metadata dicts (must include page_number, chunk_id).
        """
        if not ids:
            return
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {k: v for k, v in m.items() if v is not None}
                for m in metadatas
            ],
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        n_results: Optional[int] = None,
    ) -> dict:
        """
        Query the collection by embedding.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            n_results: Override for ChromaDB n_results (defaults to top_k).

        Returns:
            Dict with keys: ids, documents, metadatas, distances.
        """
        n = n_results or top_k
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, self._collection.count()),
        )
        return result
