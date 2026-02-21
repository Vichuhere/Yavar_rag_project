#!/usr/bin/env python3
"""
Offline RAG system - CLI entry point.
Indexes a PDF and runs an interactive Q&A loop.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from src.chunker import chunk_text
from src.embedder import Embedder
from src.generator import format_context, generate
from src.pdf_loader import load_pdf
from src.retriever import Retriever
from src.vector_db import VectorDB


def index_document(pdf_path: Path) -> None:
    """Load PDF, chunk, embed, and store in vector DB."""
    print("Loading PDF...")
    pages = list(load_pdf(pdf_path))
    if not pages:
        raise ValueError("PDF contains no text (all pages empty).")

    print("Chunking text...")
    chunks = list(
        chunk_text(
            iter(pages),
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
    )
    if not chunks:
        raise ValueError("No valid chunks produced from PDF.")

    print("Initializing embedder and vector DB...")
    embedder = Embedder()
    db = VectorDB(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    db.clear()

    print("Generating embeddings...")
    texts = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]
    embeddings = embedder.encode(texts)

    ids = [f"chunk_{i+1}" for i in range(len(chunks))]
    db.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {len(chunks)} chunks. Ready for questions.\n")


def run_interactive(pdf_path: Path) -> None:
    """Index document and enter interactive Q&A loop."""
    index_document(pdf_path)

    embedder = Embedder()
    db = VectorDB(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    retriever = Retriever(
        vector_db=db,
        embedder=embedder,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    print("Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye.")
            break

        try:
            chunks = retriever.retrieve(question)
            context = format_context(chunks)
            answer = generate(question, context)
            print("\nAnswer:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            print()
        except Exception as e:
            print(f"Error: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline RAG: question-answering from a PDF document."
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF document (30-50 pages recommended)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run questions from evaluation/test_questions.txt and print answers",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path.resolve()
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print("Error: File must be a PDF.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.eval:
            run_evaluation(pdf_path)
        else:
            run_interactive(pdf_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def run_evaluation(pdf_path: Path) -> None:
    """Run questions from evaluation/test_questions.txt."""
    eval_file = Path(__file__).parent / "evaluation" / "test_questions.txt"
    if not eval_file.exists():
        print(f"Evaluation file not found: {eval_file}", file=sys.stderr)
        sys.exit(1)

    lines = eval_file.read_text(encoding="utf-8").strip().splitlines()
    questions = [q.strip() for q in lines if q.strip() and not q.strip().startswith("#")]

    if not questions:
        print("No questions found in evaluation/test_questions.txt")
        return

    index_document(pdf_path)
    embedder = Embedder()
    db = VectorDB(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    retriever = Retriever(
        vector_db=db,
        embedder=embedder,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    for i, question in enumerate(questions, 1):
        print(f"\n[Q{i}] {question}")
        try:
            chunks = retriever.retrieve(question)
            context = format_context(chunks)
            answer = generate(question, context)
            print(f"[A{i}] {answer}")
        except Exception as e:
            print(f"[A{i}] Error: {e}")


if __name__ == "__main__":
    main()
