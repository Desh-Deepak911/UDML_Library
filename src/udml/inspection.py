"""Draw stored UDML geometry onto a diagnostic PDF copy."""

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import pymupdf

from udml.core.models import Document, Page
from udml.parsers.base import InvalidPDFError, PasswordProtectedPDFError
from udml.serialization import load_document
from udml.structure.tables import (
    RuledGridCandidate,
    RuledGridConfig,
    ruled_grid_candidates,
)

_DIMENSION_TOLERANCE = 0.01
_RECT_COLOR = (1, 0, 0)
_RECT_WIDTH = 0.75
_VECTOR_WIDTH = 0.75
_STROKED_SEGMENT_COLOR = (0, 0, 1)
_FILLED_EDGE_COLOR = (0.9, 0.45, 0)
_FILLED_EDGE_DASHES = "[2 1.5] 0"
_CELL_COLOR = (0.1, 0.55, 0.2)
_CELL_WIDTH = 1.0
_UNASSIGNED_SPAN_COLOR = (0.75, 0.0, 0.7)
_UNASSIGNED_SPAN_WIDTH = 1.0
_UNASSIGNED_SPAN_DASHES = "[3 2] 0"


class InspectionError(Exception):
    """The diagnostic copy could not be created."""


class InspectionCounts(NamedTuple):
    """Counts from one diagnostic copy."""

    pages: int
    spans: int
    segments: int
    candidates: int
    cells: int
    assigned_spans: int
    unassigned_spans: int


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
    *,
    vectors: bool = False,
    grids: RuledGridConfig | None = None,
) -> InspectionCounts:
    """Write a PDF copy with rectangles around the model's stored spans.

    When ``vectors`` is true, also stroke the model's stored vector segments.
    When ``grids`` is set, draw ruled-grid cell boxes from that configuration.
    The source PDF and UDML JSON are not modified. Text is not extracted again,
    and ``get_drawings()`` is not called.
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

    candidates: list[RuledGridCandidate] = []
    try:
        if pdf.needs_pass or pdf.is_encrypted:
            raise PasswordProtectedPDFError(
                f"{source.name} is password-protected and cannot be inspected."
            )
        if not pdf.is_pdf:
            raise InvalidPDFError(f"{source.name} is not a PDF.")
        _check_pages(pdf, document)
        if grids is not None:
            candidates = ruled_grid_candidates(document, grids)
        _draw_spans(pdf, document)
        if vectors:
            _draw_vectors(pdf, document)
        if grids is not None:
            _draw_grids(pdf, document, candidates)
        _publish_pdf(pdf, destination, source, model)
    finally:
        if not pdf.is_closed:
            pdf.close()

    span_count = sum(len(page.spans) for page in document.pages)
    segment_count = sum(len(page.vector_segments) for page in document.pages)
    candidate_count, cell_count, assigned_count, unassigned_count = _count_grids(
        candidates
    )
    return InspectionCounts(
        len(document.pages),
        span_count,
        segment_count,
        candidate_count,
        cell_count,
        assigned_count,
        unassigned_count,
    )


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


def _draw_vectors(pdf: pymupdf.Document, document: Document) -> None:
    pages = {page.number: page for page in document.pages}
    for index in range(pdf.page_count):
        segments = pages[index + 1].vector_segments
        if not segments:
            continue
        # draw_line uses the same Shape matrix as draw_rect. On a rotated
        # page that is the unrotated CropBox y-flip, not page.rotation_matrix.
        shape = pdf[index].new_shape()
        for segment in segments:
            shape.draw_line(
                pymupdf.Point(segment.start.x, segment.start.y),
                pymupdf.Point(segment.end.x, segment.end.y),
            )
            # Fixed diagnostic stroke. closePath would add a segment.
            if segment.stroked:
                shape.finish(
                    color=_STROKED_SEGMENT_COLOR,
                    width=_VECTOR_WIDTH,
                    closePath=False,
                )
            else:
                shape.finish(
                    color=_FILLED_EDGE_COLOR,
                    width=_VECTOR_WIDTH,
                    dashes=_FILLED_EDGE_DASHES,
                    closePath=False,
                )
        shape.commit(overlay=True)


def _draw_grids(
    pdf: pymupdf.Document,
    document: Document,
    candidates: list[RuledGridCandidate],
) -> None:
    """Draw supported cells and unassigned spans.

    The candidate union rectangle is not drawn. It can cover holes and is
    not a ruled boundary.
    """
    by_page: dict[int, list[RuledGridCandidate]] = {}
    for candidate in candidates:
        by_page.setdefault(candidate.page_number, []).append(candidate)
    pages = {page.number: page for page in document.pages}
    for index in range(pdf.page_count):
        page_candidates = by_page.get(index + 1)
        if not page_candidates:
            continue
        model_page = pages[index + 1]
        spans = {span.id: span for span in model_page.spans}
        # Same Shape matrix as span rectangles: unrotated CropBox y-flip.
        shape = pdf[index].new_shape()
        for candidate in page_candidates:
            for cell in candidate.cells:
                box = cell.bbox
                shape.draw_rect(pymupdf.Rect(box.x0, box.y0, box.x1, box.y1))
        shape.finish(color=_CELL_COLOR, width=_CELL_WIDTH)
        drawn: set[str] = set()
        for candidate in page_candidates:
            for span_id in candidate.unassigned_span_ids:
                if span_id in drawn:
                    continue
                drawn.add(span_id)
                span = spans.get(span_id)
                if span is None:
                    raise InspectionError(
                        f"Page {model_page.number} grid candidate references "
                        f"unknown span {span_id}."
                    )
                box = span.bbox
                shape.draw_rect(pymupdf.Rect(box.x0, box.y0, box.x1, box.y1))
        if drawn:
            shape.finish(
                color=_UNASSIGNED_SPAN_COLOR,
                width=_UNASSIGNED_SPAN_WIDTH,
                dashes=_UNASSIGNED_SPAN_DASHES,
            )
        shape.commit(overlay=True)


def _count_grids(
    candidates: list[RuledGridCandidate],
) -> tuple[int, int, int, int]:
    """Count candidates, cells, and spans once per page."""
    by_page: dict[int, list[RuledGridCandidate]] = {}
    for candidate in candidates:
        by_page.setdefault(candidate.page_number, []).append(candidate)
    cells = 0
    assigned = 0
    unassigned = 0
    for page_candidates in by_page.values():
        assigned_ids: set[str] = set()
        unassigned_ids: set[str] = set()
        for candidate in page_candidates:
            cells += len(candidate.cells)
            for cell in candidate.cells:
                assigned_ids.update(cell.span_ids)
            unassigned_ids.update(candidate.unassigned_span_ids)
        assigned += len(assigned_ids)
        unassigned += len(unassigned_ids)
    return len(candidates), cells, assigned, unassigned


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
