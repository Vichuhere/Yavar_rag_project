"""
PDF document loader using PyMuPDF (fitz).
Extracts text page-wise and skips empty pages.
"""

from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF


def load_pdf(path: str | Path) -> Generator[tuple[int, str], None, None]:
    """
    Load a PDF file and yield (page_number, text) for each non-empty page.

    Args:
        path: Path to the PDF file.

    Yields:
        Tuples of (page_number (1-based), page_text).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    doc = fitz.open(path)
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text = text.strip()
            if text:
                # Use 1-based page numbers for user-facing citations
                yield page_num + 1, text
    finally:
        doc.close()
