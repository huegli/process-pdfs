"""Process scanned PDFs and extract structured information using LLMs."""

__version__ = "0.1.0"

from .cli import (
    extract_text_from_pdf,
    analyze_document,
    categorize,
    process_pdfs,
    parse_date_to_yyyymmdd,
    create_suggested_filename,
)

__all__ = [
    "extract_text_from_pdf",
    "analyze_document",
    "categorize",
    "process_pdfs",
    "parse_date_to_yyyymmdd",
    "create_suggested_filename",
]
