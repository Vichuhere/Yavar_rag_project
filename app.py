#!/usr/bin/env python3
"""
Web UI for the offline RAG system.
Upload PDF, index, and ask questions via a simple browser interface.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DATA_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from src.chunker import chunk_text
from src.embedder import Embedder
from src.generator import format_context, generate
from src.pdf_loader import load_pdf
from src.retriever import Retriever
from src.vector_db import VectorDB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max PDF
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global retriever (set after successful indexing)
retriever: Retriever | None = None


def index_document(pdf_path: Path) -> int:
    """Index PDF and return number of chunks."""
    pages = list(load_pdf(pdf_path))
    if not pages:
        raise ValueError("PDF contains no text (all pages empty).")

    chunks = list(
        chunk_text(iter(pages), chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    )
    if not chunks:
        raise ValueError("No valid chunks produced from PDF.")

    embedder = Embedder()
    db = VectorDB(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    db.clear()

    texts = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]
    embeddings = embedder.encode(texts)
    ids = [f"chunk_{i+1}" for i in range(len(chunks))]
    db.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    global retriever
    retriever = Retriever(
        vector_db=db,
        embedder=embedder,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    return len(chunks)


@app.route("/")
def index():
    """Serve the main page with upload and Q&A UI."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle PDF upload and indexing."""
    if "pdf" not in request.files:
        return jsonify({"success": False, "error": "No file selected"}), 400

    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "File must be a PDF"}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR / filename
    file.save(str(filepath))

    try:
        num_chunks = index_document(filepath)
        return jsonify({
            "success": True,
            "message": f"Indexed {num_chunks} chunks. Ready for questions.",
            "filename": filename,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if filepath.exists():
            filepath.unlink()  # Remove after indexing


@app.route("/ask", methods=["POST"])
def ask():
    """Handle question and return answer."""
    global retriever
    if retriever is None:
        return jsonify({"error": "Please upload a PDF first"}), 400

    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        chunks = retriever.retrieve(question)
        context = format_context(chunks)
        answer = generate(question, context)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting RAG web server at http://127.0.0.1:5000")
    print("Open this URL in your browser to upload PDFs and ask questions.")
    app.run(host="127.0.0.1", port=5000, debug=False)
