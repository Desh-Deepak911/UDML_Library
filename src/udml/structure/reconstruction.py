"""Geometric rows and columns from ruled-grid candidates.

This reads stored grid cells and assignment records. It does not reopen a
PDF, re-extract text, or modify the document. A result is not a confirmed
semantic table, a header layout, or a reliable reading order.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from udml.core.geometry import BoundingBox, Point
from udml.core.models import Document, Page, TextLine, TextSpan
from udml.structure.tables import (
    RuledGridCandidate,
    RuledGridConfig,
    SpanCellAssignment,
)

TABLE_RECONSTRUCTION_V1: Literal["table-reconstruction-v1"] = "table-reconstruction-v1"
LOCAL_GEOMETRIC_ORDER: Literal["local-geometric"] = "local-geometric"
UNRESOLVED_ORDER: Literal["unresolved"] = "unresolved"
# Same numerical test as the baseline fallback. Direction is not normalized.
_HORIZONTAL_DIRECTION_TOLERANCE = 1e-3
# Shared edges are not interior overlaps.
_OVERLAP_EPSILON = 1e-6

_WarningCode = Literal[
    "overlapping-cells",
    "non-elementary-cell",
    "conflicting-cell-slot",
    "conflicting-membership",
    "unmatched-assignment",
    "assignment-also-unassigned",
    "missing-page",
    "missing-span",
    "missing-line",
    "ordering-unresolved",
]
_AssignmentMethod = Literal["bbox-containment", "baseline-fallback"]

TABLE_RECONSTRUCTION_LIMITATIONS = (
    "These are geometric reconstructions of ruled-grid candidates, "
    "not confirmed tables.",
    "Local geometric order is not a reliable reading order.",
    "Merged cells, headers, and financial meaning are not inferred.",
    "A missing grid location is omitted. An empty supported cell is kept.",
    "A span with conflicting assignment records is left unresolved.",
    "A baseline assignment does not show that glyph ink lies inside the cell.",
)


class StructuralWarning(BaseModel):
    """One structural problem found while reconstructing a grid.

    ``subject`` identifies the cell box, span, or line involved. A warning
    does not repair the geometry.
    """

    model_config = ConfigDict(extra="forbid")

    code: _WarningCode
    subject: str


class MembershipCandidate(BaseModel):
    """One source assignment that did not become resolved cell content.

    ``cell`` is the candidate cell named by the assignment record. ``bbox``
    and ``origin`` are that record's source span values.
    """

    model_config = ConfigDict(extra="forbid")

    cell: BoundingBox
    method: _AssignmentMethod
    bbox: BoundingBox
    origin: Point


class UnresolvedMembership(BaseModel):
    """A span whose assignment records do not agree.

    The span is absent from resolved cell content. Identical copies of one
    record are not listed here.
    """

    model_config = ConfigDict(extra="forbid")

    span_id: str
    candidates: list[MembershipCandidate]


class TableSpanAssignment(BaseModel):
    """One span placed in a reconstructed cell.

    ``text`` is the source span text. It is missing when the document has no
    span with ``span_id``. Text is not normalized or joined with other spans.
    """

    model_config = ConfigDict(extra="forbid")

    span_id: str
    text: str | None
    method: _AssignmentMethod


class TableCellLine(BaseModel):
    """Spans from one source line that were assigned to one cell.

    ``order`` is ``local-geometric`` only for writing mode 0 and direction
    approximately ``(1, 0)``. Other orientations stay in source order and use
    ``unresolved``. Neither value is a reliable reading order.
    """

    model_config = ConfigDict(extra="forbid")

    line_id: str
    direction: tuple[FiniteFloat, FiniteFloat]
    writing_mode: Literal[0, 1]
    order: Literal["local-geometric", "unresolved"]
    spans: list[TableSpanAssignment]


class TableCell(BaseModel):
    """One supported grid cell with geometric row and column indices.

    Indices are positions on the elementary boundary lattice: row 0 is the
    smallest y interval and column 0 is the smallest x interval. They are
    absent when the cell overlaps another cell, shares a slot, or does not
    occupy exactly one adjacent boundary interval. ``assignments`` is empty
    when the cell is supported but has no assigned text.
    """

    model_config = ConfigDict(extra="forbid")

    row: Annotated[int, Field(ge=0)] | None
    column: Annotated[int, Field(ge=0)] | None
    bbox: BoundingBox
    assignments: list[TableSpanAssignment]
    lines: list[TableCellLine]


class TableCandidate(BaseModel):
    """One ruled-grid candidate reconstructed as geometric rows and columns.

    ``id`` is the source grid candidate id. ``x_boundaries`` and
    ``y_boundaries`` are the ordered edges of cells that received indices.
    A supported cell with no text is present. An unsupported location between
    those boundaries is omitted. ``assignments`` on each cell are resolved
    memberships. ``unresolved_memberships`` keeps spans whose records disagree.
    ``unassigned_span_ids`` preserves the source list, including spans that
    also have assignment records.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    page_number: Annotated[int, Field(ge=1)]
    bbox: BoundingBox
    method: Literal["table-reconstruction-v1"] = TABLE_RECONSTRUCTION_V1
    configuration: RuledGridConfig
    x_boundaries: list[float]
    y_boundaries: list[float]
    cells: list[TableCell]
    unassigned_span_ids: list[str]
    unresolved_memberships: list[UnresolvedMembership]
    warnings: list[StructuralWarning]


def reconstructed_tables(
    document: Document,
    candidates: list[RuledGridCandidate],
) -> list[TableCandidate]:
    """Return derived table candidates for ``candidates``.

    ``document`` and ``candidates`` are not modified. See
    ``TABLE_RECONSTRUCTION_LIMITATIONS``.
    """
    pages = {page.number: page for page in document.pages}
    return [
        _reconstruct(pages.get(candidate.page_number), candidate)
        for candidate in candidates
    ]


def _reconstruct(
    page: Page | None,
    candidate: RuledGridCandidate,
) -> TableCandidate:
    warnings: list[StructuralWarning] = []
    boxes = [cell.bbox.model_copy() for cell in candidate.cells]
    indexed, kinds = _index_cells(boxes)
    for index, kind in enumerate(kinds):
        if kind is not None:
            warnings.append(
                StructuralWarning(code=kind, subject=_box_subject(boxes[index]))
            )
    x_boundaries, y_boundaries = _boundaries(
        [boxes[index] for index in range(len(boxes)) if indexed[index] is not None]
    )
    placed = _place(indexed, x_boundaries, y_boundaries, boxes, warnings)
    if page is None:
        warnings.append(
            StructuralWarning(code="missing-page", subject=str(candidate.page_number))
        )
    assignments, unresolved = _assignments(page, candidate, boxes, warnings)
    cells = [
        _cell(page, boxes[index], placed[index], assignments[index], warnings)
        for index in range(len(boxes))
    ]
    cells.sort(key=_cell_key)
    recorded_ids = {record.span_id for record in candidate.assignments}
    for span_id in candidate.unassigned_span_ids:
        if span_id in recorded_ids:
            warnings.append(
                StructuralWarning(code="assignment-also-unassigned", subject=span_id)
            )
    return TableCandidate(
        id=candidate.id,
        page_number=candidate.page_number,
        bbox=candidate.bbox.model_copy(),
        method=TABLE_RECONSTRUCTION_V1,
        configuration=candidate.configuration.model_copy(),
        x_boundaries=x_boundaries,
        y_boundaries=y_boundaries,
        cells=cells,
        unassigned_span_ids=list(candidate.unassigned_span_ids),
        unresolved_memberships=unresolved,
        warnings=warnings,
    )


def _index_cells(
    boxes: list[BoundingBox],
) -> tuple[list[tuple[int, int] | None], list[_WarningCode | None]]:
    """Return a provisional slot and a geometry warning code for each cell."""
    overlapping = _overlapping_indexes(boxes)
    usable = [index for index in range(len(boxes)) if index not in overlapping]
    x_boundaries, y_boundaries = _boundaries([boxes[index] for index in usable])
    slots: dict[tuple[int, int], list[int]] = {}
    provisional: list[tuple[int, int] | None] = [None] * len(boxes)
    kinds: list[_WarningCode | None] = [None] * len(boxes)
    for index in usable:
        slot = _slot(boxes[index], x_boundaries, y_boundaries)
        if slot is None:
            kinds[index] = "non-elementary-cell"
            continue
        provisional[index] = slot
        slots.setdefault(slot, []).append(index)
    for indexes in slots.values():
        if len(indexes) > 1:
            for index in indexes:
                provisional[index] = None
                kinds[index] = "conflicting-cell-slot"
    for index in overlapping:
        kinds[index] = "overlapping-cells"
    return provisional, kinds


def _place(
    provisional: list[tuple[int, int] | None],
    x_boundaries: list[float],
    y_boundaries: list[float],
    boxes: list[BoundingBox],
    warnings: list[StructuralWarning],
) -> list[tuple[int, int] | None]:
    """Recompute indices on the lattice of cells that already have a slot."""
    placed: list[tuple[int, int] | None] = [None] * len(boxes)
    owners: dict[tuple[int, int], int] = {}
    collisions: set[int] = set()
    for index, slot in enumerate(provisional):
        if slot is None:
            continue
        refined = _slot(boxes[index], x_boundaries, y_boundaries)
        if refined is None:
            warnings.append(
                StructuralWarning(
                    code="non-elementary-cell",
                    subject=_box_subject(boxes[index]),
                )
            )
            continue
        previous = owners.get(refined)
        if previous is not None:
            collisions.add(previous)
            collisions.add(index)
        owners[refined] = index
        placed[index] = refined
    for index in sorted(collisions):
        placed[index] = None
        warnings.append(
            StructuralWarning(
                code="conflicting-cell-slot",
                subject=_box_subject(boxes[index]),
            )
        )
    return placed


def _assignments(
    page: Page | None,
    candidate: RuledGridCandidate,
    boxes: list[BoundingBox],
    warnings: list[StructuralWarning],
) -> tuple[list[list[TableSpanAssignment]], list[UnresolvedMembership]]:
    spans = {} if page is None else {span.id: span for span in page.spans}
    grouped: list[list[TableSpanAssignment]] = [[] for _ in boxes]
    unresolved: list[UnresolvedMembership] = []
    for span_id, records in _unique_records(candidate.assignments).items():
        if len(records) > 1:
            warnings.append(
                StructuralWarning(code="conflicting-membership", subject=span_id)
            )
            unresolved.append(_unresolved(records))
            continue
        record = records[0]
        matches = [
            index for index, box in enumerate(boxes) if _same_box(box, record.cell)
        ]
        if not matches:
            warnings.append(
                StructuralWarning(code="unmatched-assignment", subject=record.span_id)
            )
            continue
        grouped[matches[0]].append(
            _assignment(record, spans.get(record.span_id), warnings)
        )
    for items in grouped:
        items.sort(key=lambda item: item.span_id)
    return grouped, unresolved


def _unique_records(
    records: list[SpanCellAssignment],
) -> dict[str, list[SpanCellAssignment]]:
    """Drop exact duplicate records and group the rest by span id."""
    grouped: dict[str, list[SpanCellAssignment]] = {}
    seen: set[tuple[object, ...]] = set()
    for record in records:
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(record.span_id, []).append(record)
    return grouped


def _record_key(record: SpanCellAssignment) -> tuple[object, ...]:
    return (
        record.span_id,
        record.method,
        record.cell.x0,
        record.cell.y0,
        record.cell.x1,
        record.cell.y1,
        record.bbox.x0,
        record.bbox.y0,
        record.bbox.x1,
        record.bbox.y1,
        record.origin.x,
        record.origin.y,
    )


def _unresolved(records: list[SpanCellAssignment]) -> UnresolvedMembership:
    return UnresolvedMembership(
        span_id=records[0].span_id,
        candidates=[
            MembershipCandidate(
                cell=record.cell.model_copy(),
                method=record.method,
                bbox=record.bbox.model_copy(),
                origin=record.origin.model_copy(),
            )
            for record in records
        ],
    )


def _assignment(
    record: SpanCellAssignment,
    span: TextSpan | None,
    warnings: list[StructuralWarning],
) -> TableSpanAssignment:
    if span is None:
        warnings.append(StructuralWarning(code="missing-span", subject=record.span_id))
    return TableSpanAssignment(
        span_id=record.span_id,
        text=None if span is None else span.text,
        method=record.method,
    )


def _cell(
    page: Page | None,
    box: BoundingBox,
    slot: tuple[int, int] | None,
    assignments: list[TableSpanAssignment],
    warnings: list[StructuralWarning],
) -> TableCell:
    row, column = (None, None) if slot is None else slot
    return TableCell(
        row=row,
        column=column,
        bbox=box,
        assignments=assignments,
        lines=_lines(page, assignments, warnings),
    )


def _lines(
    page: Page | None,
    assignments: list[TableSpanAssignment],
    warnings: list[StructuralWarning],
) -> list[TableCellLine]:
    if page is None or not assignments:
        return []
    by_span = {span.id: span for span in page.spans}
    line_by_span = {span_id: line for line in page.lines for span_id in line.span_ids}
    grouped: dict[str, list[TableSpanAssignment]] = {}
    order: list[str] = []
    for item in assignments:
        if item.span_id not in by_span:
            continue
        line = line_by_span.get(item.span_id)
        if line is None:
            warnings.append(
                StructuralWarning(code="missing-line", subject=item.span_id)
            )
            continue
        if line.id not in grouped:
            grouped[line.id] = []
            order.append(line.id)
        grouped[line.id].append(item)
    lines_by_id = {line.id: line for line in page.lines}
    geometric: list[TableCellLine] = []
    unresolved: list[TableCellLine] = []
    for line_id in order:
        line = lines_by_id[line_id]
        built = _content_line(line, grouped[line_id], by_span)
        if built.order == LOCAL_GEOMETRIC_ORDER:
            geometric.append(built)
        else:
            warnings.append(
                StructuralWarning(code="ordering-unresolved", subject=line.id)
            )
            unresolved.append(built)
    geometric.sort(key=lambda item: _line_key(item, by_span))
    return geometric + unresolved


def _content_line(
    line: TextLine,
    items: list[TableSpanAssignment],
    by_span: dict[str, TextSpan],
) -> TableCellLine:
    if _is_horizontal(line):
        ordered = sorted(items, key=lambda item: _span_key(by_span[item.span_id]))
        order: Literal["local-geometric", "unresolved"] = LOCAL_GEOMETRIC_ORDER
    else:
        source_order = {span_id: index for index, span_id in enumerate(line.span_ids)}
        ordered = sorted(items, key=lambda item: source_order[item.span_id])
        order = UNRESOLVED_ORDER
    return TableCellLine(
        line_id=line.id,
        direction=(line.direction[0], line.direction[1]),
        writing_mode=line.writing_mode,
        order=order,
        spans=ordered,
    )


def _is_horizontal(line: TextLine) -> bool:
    if line.writing_mode != 0:
        return False
    dx, dy = line.direction
    return (
        abs(dx - 1) <= _HORIZONTAL_DIRECTION_TOLERANCE
        and abs(dy) <= _HORIZONTAL_DIRECTION_TOLERANCE
    )


def _span_key(span: TextSpan) -> tuple[float, float, str]:
    return (span.origin.x, span.bbox.x0, span.id)


def _line_key(
    line: TableCellLine,
    by_span: dict[str, TextSpan],
) -> tuple[float, float, str]:
    spans = [by_span[item.span_id] for item in line.spans]
    return (
        min(span.origin.y for span in spans),
        min(span.origin.x for span in spans),
        line.line_id,
    )


def _overlapping_indexes(boxes: list[BoundingBox]) -> set[int]:
    found: set[int] = set()
    for left_index, left in enumerate(boxes):
        for right_index in range(left_index + 1, len(boxes)):
            if _overlaps(left, boxes[right_index]):
                found.add(left_index)
                found.add(right_index)
    return found


def _overlaps(left: BoundingBox, right: BoundingBox) -> bool:
    return (
        left.x0 < right.x1 - _OVERLAP_EPSILON
        and right.x0 < left.x1 - _OVERLAP_EPSILON
        and left.y0 < right.y1 - _OVERLAP_EPSILON
        and right.y0 < left.y1 - _OVERLAP_EPSILON
    )


def _boundaries(boxes: list[BoundingBox]) -> tuple[list[float], list[float]]:
    if not boxes:
        return [], []
    return (
        sorted({edge for box in boxes for edge in (box.x0, box.x1)}),
        sorted({edge for box in boxes for edge in (box.y0, box.y1)}),
    )


def _slot(
    box: BoundingBox,
    x_boundaries: list[float],
    y_boundaries: list[float],
) -> tuple[int, int] | None:
    column = _adjacent_index(x_boundaries, box.x0, box.x1)
    row = _adjacent_index(y_boundaries, box.y0, box.y1)
    if row is None or column is None:
        return None
    return row, column


def _adjacent_index(boundaries: list[float], start: float, end: float) -> int | None:
    for index in range(len(boundaries) - 1):
        if boundaries[index] == start and boundaries[index + 1] == end:
            return index
    return None


def _same_box(left: BoundingBox, right: BoundingBox) -> bool:
    return (left.x0, left.y0, left.x1, left.y1) == (right.x0, right.y0, right.x1, right.y1)


def _box_subject(box: BoundingBox) -> str:
    return f"{box.x0},{box.y0},{box.x1},{box.y1}"


def _cell_key(cell: TableCell) -> tuple[bool, int, int, float, float, float, float]:
    unindexed = cell.row is None or cell.column is None
    return (
        unindexed,
        0 if cell.row is None else cell.row,
        0 if cell.column is None else cell.column,
        cell.bbox.y0,
        cell.bbox.x0,
        cell.bbox.y1,
        cell.bbox.x1,
    )
