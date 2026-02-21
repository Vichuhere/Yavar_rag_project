"""Create a minimal test PDF for running the RAG system."""
import fitz

doc = fitz.open()
for i in range(5):
    page = doc.new_page()
    text = (
        f"Page {i+1}. This is sample content for the offline RAG system. "
        "The document discusses retrieval-augmented generation. "
        "Key points: use local LLM via Ollama, ChromaDB for vector storage, "
        "sentence-transformers for embeddings. The main recommendation is to "
        "run everything offline without cloud services. "
    ) * 15
    page.insert_text((50, 50), text)
doc.save("data/sample_document.pdf")
doc.close()
print("Created data/sample_document.pdf")
