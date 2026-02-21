"""
Configuration for the offline RAG system.
All parameters are centralized here - no hardcoded values in logic files.
"""

import os
from pathlib import Path

# Load .env for GROQ_API_KEY (optional)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------
# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# sentence-transformers model for local embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Batch size for encoding (improves throughput)
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Vector Database (ChromaDB)
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = "grounded_rag_docs"
# Persist to local disk
CHROMA_PERSIST_DIR = str(VECTOR_STORE_DIR)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 4
# Similarity threshold - chunks below this are discarded
SIMILARITY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Generation - Groq API (preferred when API key is set)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, good for RAG

# ---------------------------------------------------------------------------
# Generation - Ollama (fallback when no Groq API key)
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TEMPERATURE = 0.0
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# Fallback message when answer not found in document
# ---------------------------------------------------------------------------
FALLBACK_MESSAGE = "The information is not available in the document."
