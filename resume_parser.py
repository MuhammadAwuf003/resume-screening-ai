"""
resume_parser.py
----------------
Handles PDF text extraction using PyPDF2.
Provides a clean interface for reading resume content from uploaded files.
"""

import PyPDF2
import io


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract all text from an uploaded PDF file object.

    Args:
        pdf_file: A file-like object (e.g., Streamlit UploadedFile or BytesIO).

    Returns:
        A single string containing all extracted text from the PDF,
        with pages separated by newlines. Returns empty string on failure.
    """
    text = ""
    try:
        # Wrap in BytesIO if needed (Streamlit UploadedFile is already file-like)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))

        # Iterate over every page and accumulate text
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except PyPDF2.errors.PdfReadError as e:
        # Corrupted or encrypted PDFs will raise this
        raise ValueError(f"Could not read PDF file: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error while parsing PDF: {e}")

    return text.strip()


def get_pdf_metadata(pdf_file) -> dict:
    """
    Extract basic metadata from a PDF (author, title, pages, etc.).

    Args:
        pdf_file: A file-like object.

    Returns:
        A dictionary with metadata fields. Missing fields default to 'N/A'.
    """
    metadata = {
        "title": "N/A",
        "author": "N/A",
        "num_pages": 0,
    }
    try:
        pdf_file.seek(0)  # Reset pointer before re-reading
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        info = pdf_reader.metadata or {}
        metadata["title"] = info.get("/Title", "N/A")
        metadata["author"] = info.get("/Author", "N/A")
        metadata["num_pages"] = len(pdf_reader.pages)
    except Exception:
        pass  # Metadata is non-critical; silently skip errors

    return metadata
