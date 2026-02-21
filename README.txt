OFFLINE RAG - PDF Q&A SYSTEM
============================

What this project does:
- Accepts a PDF document (30-50 pages).
- Answers questions strictly from the document.
- Provides page number citations.
- Uses Groq API (when API key in .env) or Ollama (local fallback).
- Uses sentence-transformers + ChromaDB for embeddings.

How to run (with Groq API):
1. Install: pip install -r requirements.txt
2. API key is in .env (Groq) - no Ollama needed
3. CLI: python main.py path\to\your.pdf
4. Web UI: python app.py then open http://127.0.0.1:5000

How to run (with Ollama instead):
- Remove or empty GROQ_API_KEY in .env
- Run: ollama pull llama3.2:3b
