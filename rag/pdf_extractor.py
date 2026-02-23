"""
pdf_extractor.py
----------------
Responsible for extracting raw text from a PDF file.
"""

import pdfplumber


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF, annotating each page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full extracted text with page markers.
    """
    all_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            all_text += f"\n--- Page {page_number + 1} ---\n"
            all_text += page_text
            all_text += "\n"

    return all_text
