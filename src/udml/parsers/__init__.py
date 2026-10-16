"""PDF text-span parsers."""

from udml.parsers.base import (
    InvalidPDFError,
    PDFExtractionError,
    PDFParser,
    PasswordProtectedPDFError,
)
from udml.parsers.pymupdf_parser import PyMuPDFParser

__all__ = [
    "InvalidPDFError",
    "PDFExtractionError",
    "PDFParser",
    "PasswordProtectedPDFError",
    "PyMuPDFParser",
]
