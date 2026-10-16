"""Parser interface for PDF text extraction."""

from pathlib import Path
from typing import Protocol

from udml.core.models import Document


class PDFExtractionError(Exception):
    """Extraction failed, so no document is returned."""


class InvalidPDFError(PDFExtractionError):
    """The file is not a readable PDF."""


class PasswordProtectedPDFError(PDFExtractionError):
    """The PDF requires a password, so it was not extracted."""


class PDFParser(Protocol):
    """Extracts physical text from a PDF into a Document."""

    def parse(self, path: str | Path) -> Document:
        """Return the extracted document for the PDF at path."""
