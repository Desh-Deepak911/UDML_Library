"""Extracted physical text from a PDF.

These models store page geometry, text runs, straight vector segments, and
the parser's source line and text-block grouping. This is not a reliable
semantic structure or a complete preservation of the PDF. Span, line, and
block order follow the parser's source order, which is not a reliable
reading order. Vector segments are physical drawing evidence, not tables.
There are no semantic labels.
"""

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

from udml.core.geometry import BoundingBox, Point, union_boxes

SCHEMA_VERSION: Literal["0.4.0"] = "0.4.0"

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]
PositiveFiniteFloat = Annotated[FiniteFloat, Field(gt=0)]
NonNegativeFiniteFloat = Annotated[FiniteFloat, Field(ge=0)]
SourceIndex = Annotated[int, Field(ge=0)]


class TextSpan(BaseModel):
    """One physical text run and its location in the parser output.

    ``text`` keeps the extracted characters, including spaces. ``font_flags``
    is the parser's raw integer and is left undecoded. ``block_index``,
    ``line_index``, and ``span_index`` are zero-based positions in that
    source structure.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    id: str
    text: str
    bbox: BoundingBox
    font_name: str
    font_size: NonNegativeFiniteFloat
    font_flags: int
    block_index: SourceIndex
    line_index: SourceIndex
    span_index: SourceIndex


class TextLine(BaseModel):
    """Spans that share one parser source line.

    ``source_block_index`` and ``source_line_index`` are that source position.
    ``direction`` is PyMuPDF's line ``dir``: two finite numbers, stored as
    reported and not normalized. ``writing_mode`` is PyMuPDF's ``wmode``:
    0 for horizontal text and 1 for vertical text. Both are text orientation
    in the extracted coordinate space, not page rotation. ``span_ids`` lists
    spans in extraction order. ``bbox`` is their union. This is parser
    grouping, not a semantic line or a reliable reading order. Span text is
    not stored again.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source_block_index: SourceIndex
    source_line_index: SourceIndex
    direction: tuple[FiniteFloat, FiniteFloat]
    writing_mode: Literal[0, 1]
    span_ids: list[str] = Field(min_length=1)
    bbox: BoundingBox

    @model_validator(mode="after")
    def direction_is_nonzero(self) -> Self:
        x, y = self.direction
        if x == 0 and y == 0:
            raise ValueError("text direction must be nonzero")
        return self


class TextBlock(BaseModel):
    """Source lines that share one parser text block.

    ``source_block_index`` is that source position. ``line_ids`` lists those
    lines in extraction order. ``bbox`` is their union. This is parser
    grouping, not a semantic block or a reliable reading order.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source_block_index: SourceIndex
    line_ids: list[str] = Field(min_length=1)
    bbox: BoundingBox


class VectorSegment(BaseModel):
    """One straight segment from a page drawing.

    ``start`` and ``end`` use the unrotated page coordinates. They are not a
    bounding box and are not sorted. ``source_drawing_index`` and
    ``source_item_index`` are positions in PyMuPDF's ``get_drawings()``
    output. ``source_edge_index`` is 0 for a line item. A rectangle item
    produces edges 0 through 3: ``(x0, y0)`` to ``(x1, y0)``, ``(x1, y0)`` to
    ``(x1, y1)``, ``(x1, y1)`` to ``(x0, y1)``, then ``(x0, y1)`` to
    ``(x0, y0)``. ``stroked`` and ``filled`` record the path paint type.
    A filled rectangle keeps its geometric edges with ``stroked`` false when
    the path is not stroked. ``stroke_width`` is the reported width, or
    missing when the drawing does not provide one. This is not a table edge
    or a claim about visible ink.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    start: Point
    end: Point
    source_drawing_index: SourceIndex
    source_item_index: SourceIndex
    source_edge_index: SourceIndex
    stroked: bool
    filled: bool
    stroke_width: NonNegativeFiniteFloat | None = None


class Page(BaseModel):
    """One PDF page in unrotated user space.

    ``width`` and ``height`` are positive dimensions in PDF points before
    rotation. ``rotation`` is stored separately as 0, 90, 180, or 270.
    ``spans`` is the only text collection. ``lines`` and ``blocks`` reference
    those spans in parser source order. ``vector_segments`` are straight
    drawing segments in source drawing order, not a complete path or a table.
    """

    model_config = ConfigDict(extra="forbid")

    number: Annotated[int, Field(ge=1)]
    width: PositiveFiniteFloat
    height: PositiveFiniteFloat
    rotation: Literal[0, 90, 180, 270]
    spans: list[TextSpan] = Field(default_factory=list)
    lines: list[TextLine] = Field(default_factory=list)
    blocks: list[TextBlock] = Field(default_factory=list)
    vector_segments: list[VectorSegment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def source_groups_match_spans(self) -> Self:
        _validate_source_groups(self)
        return self


class Document(BaseModel):
    """Provenance and pages for one extraction.

    ``warnings`` and ``limitations`` record extraction loss. They are not
    semantic annotations.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.4.0"] = SCHEMA_VERSION
    source_filename: str
    source_sha256: Sha256Hex
    parser_name: str
    parser_version: str
    extracted_at: AwareDatetime
    pages: list[Page] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_unsupported_schema_version(cls, value: object) -> object:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {value!r}; expected {SCHEMA_VERSION}. "
                "Regenerate this JSON with parse."
            )
        return value

    @model_validator(mode="after")
    def unique_page_numbers_and_span_ids(self) -> Self:
        page_numbers = [page.number for page in self.pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("page numbers must be unique")

        span_ids = [span.id for page in self.pages for span in page.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("span identifiers must be unique within the document")
        segment_ids = [
            segment.id for page in self.pages for segment in page.vector_segments
        ]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError(
                "vector segment identifiers must be unique within the document"
            )
        return self


# PyMuPDF line ``dir`` and ``wmode`` for one source line. Direction is not
# normalized and is not page rotation.
SourceLineOrientation = tuple[tuple[float, float], int]


def build_source_groups(
    page_number: int,
    spans: list[TextSpan],
    orientations: Mapping[tuple[int, int], SourceLineOrientation],
) -> tuple[list[TextLine], list[TextBlock]]:
    """Group ``spans`` by the source indices already stored on them.

    ``orientations`` is keyed by ``(block_index, line_index)`` and must
    already contain the parser line's direction and writing mode. Lines and
    blocks follow extraction order. Spans that share a source block and line
    become one line; lines that share a source block become one text block.
    Empty groups are omitted. Identifiers are ``p{page}-b{block}-l{line}``
    and ``p{page}-b{block}``. Loading JSON does not call this.
    """
    spans_by_key: dict[tuple[int, int], list[TextSpan]] = {}
    line_keys: list[tuple[int, int]] = []
    for span in spans:
        key = (span.block_index, span.line_index)
        if key not in spans_by_key:
            spans_by_key[key] = []
            line_keys.append(key)
        spans_by_key[key].append(span)

    lines: list[TextLine] = []
    lines_by_block: dict[int, list[TextLine]] = {}
    block_keys: list[int] = []
    for block_index, line_index in line_keys:
        members = spans_by_key[(block_index, line_index)]
        try:
            direction, writing_mode = orientations[(block_index, line_index)]
        except KeyError:
            raise ValueError(
                f"text orientation is missing for page {page_number}, "
                f"block {block_index}, line {line_index}"
            ) from None
        line = TextLine(
            id=f"p{page_number}-b{block_index}-l{line_index}",
            source_block_index=block_index,
            source_line_index=line_index,
            direction=direction,
            writing_mode=writing_mode,
            span_ids=[span.id for span in members],
            bbox=union_boxes([span.bbox for span in members]),
        )
        lines.append(line)
        if block_index not in lines_by_block:
            lines_by_block[block_index] = []
            block_keys.append(block_index)
        lines_by_block[block_index].append(line)

    blocks = [
        TextBlock(
            id=f"p{page_number}-b{block_index}",
            source_block_index=block_index,
            line_ids=[line.id for line in lines_by_block[block_index]],
            bbox=union_boxes([line.bbox for line in lines_by_block[block_index]]),
        )
        for block_index in block_keys
    ]
    return lines, blocks


def _require_unique(ids: list[str], message: str) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(message)


def _validate_source_groups(page: Page) -> None:
    _require_unique(
        [span.id for span in page.spans],
        "span identifiers must be unique within the page",
    )
    _require_unique(
        [line.id for line in page.lines],
        "line identifiers must be unique within the page",
    )
    _require_unique(
        [block.id for block in page.blocks],
        "block identifiers must be unique within the page",
    )
    _validate_lines(page)
    _validate_blocks(page)


def _validate_lines(page: Page) -> None:
    spans_by_id = {span.id: span for span in page.spans}
    spans_by_key: dict[tuple[int, int], list[TextSpan]] = {}
    line_keys: list[tuple[int, int]] = []
    for span in page.spans:
        key = (span.block_index, span.line_index)
        if key not in spans_by_key:
            spans_by_key[key] = []
            line_keys.append(key)
        spans_by_key[key].append(span)

    actual_keys = [
        (line.source_block_index, line.source_line_index) for line in page.lines
    ]
    if actual_keys != line_keys:
        raise ValueError(
            "source lines must follow span source indices in extraction order"
        )

    referenced: list[str] = []
    for line in page.lines:
        members: list[TextSpan] = []
        for span_id in line.span_ids:
            span = spans_by_id.get(span_id)
            if span is None:
                raise ValueError(
                    f"line {line.id} references span {span_id}, "
                    "which is not on this page"
                )
            if (
                span.block_index != line.source_block_index
                or span.line_index != line.source_line_index
            ):
                raise ValueError(
                    f"span {span_id} source indices do not match line {line.id}"
                )
            members.append(span)
        expected_ids = [
            span.id
            for span in spans_by_key[
                (line.source_block_index, line.source_line_index)
            ]
        ]
        if line.span_ids != expected_ids:
            raise ValueError(
                f"line {line.id} must reference each matching span once, "
                "in extraction order"
            )
        if line.bbox != union_boxes([span.bbox for span in members]):
            raise ValueError(f"line {line.id} box is not the union of its spans")
        referenced.extend(line.span_ids)

    if referenced != [span.id for span in page.spans]:
        raise ValueError("each span must belong to exactly one source line")


def _validate_blocks(page: Page) -> None:
    lines_by_id = {line.id: line for line in page.lines}
    lines_by_block: dict[int, list[TextLine]] = {}
    block_keys: list[int] = []
    for line in page.lines:
        if line.source_block_index not in lines_by_block:
            lines_by_block[line.source_block_index] = []
            block_keys.append(line.source_block_index)
        lines_by_block[line.source_block_index].append(line)

    actual_keys = [block.source_block_index for block in page.blocks]
    if actual_keys != block_keys:
        raise ValueError(
            "source text blocks must follow line source indices in extraction order"
        )

    referenced: list[str] = []
    for block in page.blocks:
        members: list[TextLine] = []
        for line_id in block.line_ids:
            line = lines_by_id.get(line_id)
            if line is None:
                raise ValueError(
                    f"block {block.id} references line {line_id}, "
                    "which is not on this page"
                )
            if line.source_block_index != block.source_block_index:
                raise ValueError(
                    f"line {line_id} source block index does not match "
                    f"block {block.id}"
                )
            members.append(line)
        expected_ids = [
            line.id for line in lines_by_block[block.source_block_index]
        ]
        if block.line_ids != expected_ids:
            raise ValueError(
                f"block {block.id} must reference each matching line once, "
                "in extraction order"
            )
        if block.bbox != union_boxes([line.bbox for line in members]):
            raise ValueError(f"block {block.id} box is not the union of its lines")
        referenced.extend(block.line_ids)

    if referenced != [line.id for line in page.lines]:
        raise ValueError(
            "each source line must belong to exactly one source text block"
        )
