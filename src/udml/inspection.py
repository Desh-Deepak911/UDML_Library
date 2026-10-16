"""Draw stored UDML span boxes onto a diagnostic PDF copy."""

import hashlib
import math
import os
import tempfile
from pathlib import Path

import pymupdf

from udml.core.models import Document, Page
from udml.parsers.base import InvalidPDFError, PasswordProtectedPDFError
from udml.serialization import load_document

_DIMENSION_TOLERANCE = 0.01
_RECT_COLOR = (1, 0, 0)
_RECT_WIDTH = 0.75


class InspectionError(Exception):
    """The diagnostic copy could not be created."""


def paths_refer_to_same_file(left: Path, right: Path) -> bool:
    """Return whether two paths name the same file, including aliases."""
    if left.resolve() == right.resolve():
        return True
    if not left.exists() or not right.exists():
        return False
    return left.samefile(right)


def inspect_pdf(
    pdf_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> tuple[int, int]:
    """Write a PDF copy with rectangles around the model's stored spans.

    The source PDF and UDML JSON are not modified. Text is not extracted again.
    """
    source = Path(pdf_path)
    model = Path(model_path)
    destination = Path(output_path)
    _reject_output(destination, source, model)
    if destination.exists():
        raise FileExistsError(f"Output file already exists: {destination}")
    if not source.is_file():
        raise FileNotFoundError(f"PDF file not found: {source}")

    data = source.read_bytes()
    document = load_document(model)
    digest = hashlib.sha256(data).hexdigest()
    if digest.lower() != document.source_sha256.lower():
        raise InspectionError("Source PDF SHA-256 does not match the UDML document.")

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise InvalidPDFError(f"{source.name} is not a readable PDF.") from exc

    try:
        if pdf.needs_pass or pdf.is_encrypted:
            raise PasswordProtectedPDFError(
                f"{source.name} is password-protected and cannot be inspected."
            )
        if not pdf.is_pdf:
            raise InvalidPDFError(f"{source.name} is not a PDF.")
        _check_pages(pdf, document)
        _draw_spans(pdf, document)
        _publish_pdf(pdf, destination, source, model)
    finally:
        if not pdf.is_closed:
            pdf.close()

    span_count = sum(len(page.spans) for page in document.pages)
    return len(document.pages), span_count


def _reject_output(destination: Path, *inputs: Path) -> None:
    for candidate in inputs:
        if paths_refer_to_same_file(destination, candidate):
            raise InspectionError("Output path refers to an input file.")


def _check_pages(pdf: pymupdf.Document, document: Document) -> None:
    expected = set(range(1, pdf.page_count + 1))
    actual = {page.number for page in document.pages}
    if len(document.pages) != pdf.page_count or actual != expected:
        raise InspectionError(
            f"PDF has {pdf.page_count} pages; the UDML document has "
            f"{len(document.pages)}."
        )
    pages = {page.number: page for page in document.pages}
    for index in range(pdf.page_count):
        _check_page_geometry(pdf[index], pages[index + 1])


def _check_page_geometry(pdf_page: pymupdf.Page, model_page: Page) -> None:
    number = model_page.number
    width = pdf_page.cropbox.width
    height = pdf_page.cropbox.height
    if not _close(width, model_page.width):
        raise InspectionError(
            f"Page {number} width is {width} in the PDF and "
            f"{model_page.width} in the UDML document."
        )
    if not _close(height, model_page.height):
        raise InspectionError(
            f"Page {number} height is {height} in the PDF and "
            f"{model_page.height} in the UDML document."
        )
    if pdf_page.rotation != model_page.rotation:
        raise InspectionError(
            f"Page {number} rotation is {pdf_page.rotation} in the PDF and "
            f"{model_page.rotation} in the UDML document."
        )


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=_DIMENSION_TOLERANCE)


def _draw_spans(pdf: pymupdf.Document, document: Document) -> None:
    pages = {page.number: page for page in document.pages}
    for index in range(pdf.page_count):
        model_page = pages[index + 1]
        if not model_page.spans:
            continue
        # Shape converts through page.transformation_matrix. On a rotated page
        # that matrix is the unrotated CropBox y-flip, not page.rotation_matrix.
        shape = pdf[index].new_shape()
        for span in model_page.spans:
            box = span.bbox
            shape.draw_rect(pymupdf.Rect(box.x0, box.y0, box.x1, box.y1))
        shape.finish(color=_RECT_COLOR, width=_RECT_WIDTH)
        shape.commit(overlay=True)


def _publish_pdf(
    pdf: pymupdf.Document,
    destination: Path,
    *inputs: Path,
) -> None:
    _reject_output(destination, *inputs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Output file already exists: {destination}")

    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=".udml-",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pdf.save(str(temporary))
        _reject_output(destination, *inputs)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"Output file already exists: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
