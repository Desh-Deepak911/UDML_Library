"""Extract PDF text spans with PyMuPDF.

Span order follows PyMuPDF's extraction order. It is not a reading order.
"""

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from udml.core.geometry import BoundingBox, Point
from udml.core.models import (
    Document,
    Page,
    SourceLineOrientation,
    TextSpan,
    VectorSegment,
    build_source_groups,
)
from udml.parsers.base import (
    InvalidPDFError,
    PDFExtractionError,
    PasswordProtectedPDFError,
)

_PARSER_NAME = "pymupdf"
# PyMuPDF's dict defaults, without the flag that pulls image bytes.
_TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
_TEXT_BLOCK_TYPE = pymupdf.mupdf.FZ_STEXT_BLOCK_TEXT

_NO_TEXT_WARNING = "No text spans were extracted from this page."
_REPAIR_WARNING = "PyMuPDF repaired this PDF during extraction."
_TEXT_FLAG_NAMES = (
    "TEXT_PRESERVE_LIGATURES",
    "TEXT_PRESERVE_WHITESPACE",
    "TEXT_PRESERVE_IMAGES",
    "TEXT_INHIBIT_SPACES",
    "TEXT_DEHYPHENATE",
    "TEXT_PRESERVE_SPANS",
    "TEXT_MEDIABOX_CLIP",
    "TEXT_USE_CID_FOR_UNKNOWN_UNICODE",
    "TEXT_COLLECT_STRUCTURE",
    "TEXT_ACCURATE_BBOXES",
    "TEXT_COLLECT_VECTORS",
    "TEXT_IGNORE_ACTUALTEXT",
    "TEXT_SEGMENT",
    "TEXT_CLIP",
    "TEXT_LAZY_VECTORS",
    "TEXT_FUZZY_VECTORS",
    "TEXT_PARAGRAPH_BREAK",
    "TEXT_TABLE_HUNT",
    "TEXT_COLLECT_STYLES",
    "TEXT_USE_GID_FOR_UNKNOWN_UNICODE",
    "TEXT_CLIP_RECT",
    "TEXT_ACCURATE_ASCENDERS",
    "TEXT_ACCURATE_SIDE_BEARINGS",
)


class PyMuPDFParser:
    """Build a Document from PyMuPDF structured text."""

    def parse(self, path: str | Path) -> Document:
        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        data = pdf_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        extracted_at = datetime.now(UTC)
        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except pymupdf.FileDataError as exc:
            raise InvalidPDFError(f"{pdf_path.name} is not a readable PDF.") from exc

        try:
            if pdf.needs_pass or pdf.is_encrypted:
                raise PasswordProtectedPDFError(
                    f"{pdf_path.name} is password-protected and cannot be extracted."
                )
            if not pdf.is_pdf:
                raise InvalidPDFError(f"{pdf_path.name} is not a PDF.")

            pages = [
                _convert_page(pdf[index], index + 1)
                for index in range(pdf.page_count)
            ]
            warnings = [_REPAIR_WARNING] if pdf.is_repaired else []
        finally:
            if not pdf.is_closed:
                pdf.close()

        return Document(
            source_filename=pdf_path.name,
            source_sha256=digest,
            parser_name=_PARSER_NAME,
            parser_version=str(pymupdf.VersionBind),
            extracted_at=extracted_at,
            pages=pages,
            warnings=warnings,
            limitations=_extraction_limitations(),
        )


def _describe_flags(flags: int) -> str:
    names: list[str] = []
    recognized = 0
    for name in _TEXT_FLAG_NAMES:
        bit = getattr(pymupdf, name, None)
        if not isinstance(bit, int) or bit == 0:
            continue
        recognized |= bit
        if flags & bit:
            names.append(name)
    remainder = flags & ~recognized
    if remainder:
        names.append(f"0x{remainder:x}")
    listed = ", ".join(names) if names else "none"
    return f"{listed} ({flags:#x})"


def _extraction_limitations() -> list[str]:
    limitations = [
        "No OCR is performed.",
        "Images, annotations, and semantic or table structure are not represented.",
        "Extracted text reflects PyMuPDF's decoding, not complete PDF fidelity.",
        (
            "Page width and height are the unrotated CropBox in PDF points. "
            "Span coordinates are the boxes from get_text('dict', sort=False) "
            "and are not transformed again."
        ),
        f"PyMuPDF text extraction flags: {_describe_flags(_TEXT_FLAGS)}.",
    ]
    if _TEXT_FLAGS & pymupdf.TEXT_MEDIABOX_CLIP:
        limitations.append(
            "TEXT_MEDIABOX_CLIP is enabled, so text outside the extraction "
            "bounds may be omitted."
        )
    limitations.append(
        "Vector segments are a partial representation of get_drawings(). "
        "Clipping, curves, quads, rendering appearance, and full path "
        "semantics are not preserved. Endpoints are stored as returned "
        "after PyMuPDF clears page rotation, with no further transform."
    )
    return limitations


def _convert_page(page: pymupdf.Page, number: int) -> Page:
    # CropBox is the unrotated page. page.rect swaps its axes at 90 and 270.
    rotation = page.rotation
    width = page.cropbox.width
    height = page.cropbox.height
    # get_text clears rotation while it runs, so boxes are already unrotated
    # and use a top-left origin with y increasing downward.
    text = page.get_text("dict", sort=False, flags=_TEXT_FLAGS)
    spans, orientations = _convert_spans(number, text)
    # Groups reuse the spans above. Indices and extraction order are unchanged.
    lines, blocks = build_source_groups(number, spans, orientations)
    # get_drawings clears rotation and restores it. Keep its points as returned.
    segments = _convert_segments(number, page.get_drawings())
    warnings = [] if spans else [_NO_TEXT_WARNING]
    return Page(
        number=number,
        width=width,
        height=height,
        rotation=rotation,
        spans=spans,
        lines=lines,
        blocks=blocks,
        vector_segments=segments,
        warnings=warnings,
    )


def _convert_spans(
    page_number: int, text: dict
) -> tuple[list[TextSpan], dict[tuple[int, int], SourceLineOrientation]]:
    spans: list[TextSpan] = []
    orientations: dict[tuple[int, int], SourceLineOrientation] = {}
    for block in text["blocks"]:
        if block["type"] != _TEXT_BLOCK_TYPE:
            continue
        block_index = block["number"]
        for line_index, line in enumerate(block["lines"]):
            line_spans = line["spans"]
            if line_spans:
                # dir and wmode belong to this line. Do not infer them from
                # span boxes or from page rotation.
                orientations[(block_index, line_index)] = _read_line_orientation(
                    line, page_number, block_index, line_index
                )
            for span_index, span in enumerate(line_spans):
                x0, y0, x1, y1 = span["bbox"]
                spans.append(
                    TextSpan(
                        id=f"p{page_number}-b{block_index}-l{line_index}-s{span_index}",
                        text=span["text"],
                        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                        font_name=span["font"],
                        font_size=span["size"],
                        font_flags=span["flags"],
                        block_index=block_index,
                        line_index=line_index,
                        span_index=span_index,
                    )
                )
    return spans, orientations


def _read_line_orientation(
    line: dict,
    page_number: int,
    block_index: int,
    line_index: int,
) -> SourceLineOrientation:
    """Read PyMuPDF ``dir`` and ``wmode`` without changing them."""
    where = f"page {page_number}, block {block_index}, line {line_index}"
    try:
        raw_direction = line["dir"]
        raw_mode = line["wmode"]
    except KeyError as exc:
        raise PDFExtractionError(
            f"PyMuPDF line is missing {exc.args[0]!r} ({where})."
        ) from exc
    return _direction_vector(raw_direction, where), _writing_mode(raw_mode, where)


def _direction_vector(value: object, where: str) -> tuple[float, float]:
    message = (
        "PyMuPDF line direction must contain two finite numbers "
        f"({where})."
    )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PDFExtractionError(message)
    if len(value) != 2:
        raise PDFExtractionError(message)
    components: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise PDFExtractionError(message)
        number = float(component)
        if not math.isfinite(number):
            raise PDFExtractionError(message)
        components.append(number)
    if components[0] == 0 and components[1] == 0:
        raise PDFExtractionError(
            f"PyMuPDF line direction must be nonzero ({where})."
        )
    return components[0], components[1]


def _writing_mode(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise PDFExtractionError(
            "PyMuPDF line writing mode must be 0 (horizontal) or "
            f"1 (vertical) ({where})."
        )
    return value


_PAINTED_PATHS = frozenset({"f", "s", "fs"})


def _convert_segments(page_number: int, drawings: list) -> list[VectorSegment]:
    """Keep straight lines and the four edges of rectangle items.

    Curves, quads, clips, and other path operations are skipped. A close
    flag does not add a segment. Rectangle edges use the fixed corner order
    on ``VectorSegment``. Points are not transformed.
    """
    segments: list[VectorSegment] = []
    for drawing_index, drawing in enumerate(drawings):
        where = f"page {page_number}, drawing {drawing_index}"
        if not isinstance(drawing, dict):
            raise PDFExtractionError(f"PyMuPDF drawing must be a mapping ({where}).")
        try:
            path_type = drawing["type"]
        except KeyError as exc:
            raise PDFExtractionError(
                f"PyMuPDF drawing is missing {exc.args[0]!r} ({where})."
            ) from exc
        if path_type not in _PAINTED_PATHS:
            continue
        stroked = path_type in {"s", "fs"}
        filled = path_type in {"f", "fs"}
        stroke_width = _stroke_width(drawing.get("width"), where)
        try:
            items = drawing["items"]
        except KeyError as exc:
            raise PDFExtractionError(
                f"PyMuPDF drawing is missing {exc.args[0]!r} ({where})."
            ) from exc
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise PDFExtractionError(f"PyMuPDF drawing items must be a list ({where}).")
        for item_index, item in enumerate(items):
            segments.extend(
                _item_segments(
                    page_number,
                    drawing_index,
                    item_index,
                    item,
                    stroked=stroked,
                    filled=filled,
                    stroke_width=stroke_width,
                )
            )
    return segments


def _stroke_width(value: object, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PDFExtractionError(
            f"PyMuPDF stroke width must be a nonnegative finite number ({where})."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise PDFExtractionError(
            f"PyMuPDF stroke width must be a nonnegative finite number ({where})."
        )
    return number


def _item_segments(
    page_number: int,
    drawing_index: int,
    item_index: int,
    item: object,
    *,
    stroked: bool,
    filled: bool,
    stroke_width: float | None,
) -> list[VectorSegment]:
    where = f"page {page_number}, drawing {drawing_index}, item {item_index}"
    if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or not item:
        raise PDFExtractionError(f"PyMuPDF path item is not usable ({where}).")
    kind = item[0]
    if not isinstance(kind, str):
        raise PDFExtractionError(f"PyMuPDF path item is not usable ({where}).")
    if kind == "l":
        if len(item) != 3:
            raise PDFExtractionError(
                f"PyMuPDF line item must have two points ({where})."
            )
        return [
            _make_segment(
                page_number,
                drawing_index,
                item_index,
                0,
                _point(item[1], where),
                _point(item[2], where),
                stroked,
                filled,
                stroke_width,
            )
        ]
    if kind == "re":
        if len(item) < 2:
            raise PDFExtractionError(
                f"PyMuPDF rectangle item must contain a rectangle ({where})."
            )
        return [
            _make_segment(
                page_number,
                drawing_index,
                item_index,
                edge_index,
                start,
                end,
                stroked,
                filled,
                stroke_width,
            )
            for edge_index, (start, end) in enumerate(_rectangle_edges(item[1], where))
        ]
    return []


def _make_segment(
    page_number: int,
    drawing_index: int,
    item_index: int,
    edge_index: int,
    start: Point,
    end: Point,
    stroked: bool,
    filled: bool,
    stroke_width: float | None,
) -> VectorSegment:
    return VectorSegment(
        id=f"p{page_number}-d{drawing_index}-i{item_index}-e{edge_index}",
        start=start,
        end=end,
        source_drawing_index=drawing_index,
        source_item_index=item_index,
        source_edge_index=edge_index,
        stroked=stroked,
        filled=filled,
        stroke_width=stroke_width,
    )


def _rectangle_edges(rect: object, where: str) -> list[tuple[Point, Point]]:
    x0, y0, x1, y1 = _rectangle_corners(rect, where)
    corners = (
        Point(x=x0, y=y0),
        Point(x=x1, y=y0),
        Point(x=x1, y=y1),
        Point(x=x0, y=y1),
    )
    return [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]


def _rectangle_corners(rect: object, where: str) -> tuple[float, float, float, float]:
    message = f"PyMuPDF rectangle item must contain four finite numbers ({where})."
    if all(hasattr(rect, name) for name in ("x0", "y0", "x1", "y1")):
        values = (rect.x0, rect.y0, rect.x1, rect.y1)
    elif (
        isinstance(rect, Sequence)
        and not isinstance(rect, (str, bytes))
        and len(rect) == 4
    ):
        values = tuple(rect)
    else:
        raise PDFExtractionError(message)
    numbers = tuple(_finite_number(value, message) for value in values)
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _point(value: object, where: str) -> Point:
    message = f"PyMuPDF line point must contain two finite numbers ({where})."
    if hasattr(value, "x") and hasattr(value, "y"):
        raw = (value.x, value.y)
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        raw = (value[0], value[1])
    else:
        raise PDFExtractionError(message)
    x, y = (_finite_number(component, message) for component in raw)
    return Point(x=x, y=y)


def _finite_number(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PDFExtractionError(message)
    number = float(value)
    if not math.isfinite(number):
        raise PDFExtractionError(message)
    return number
