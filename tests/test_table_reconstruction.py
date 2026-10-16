from datetime import datetime, timedelta, timezone

from udml.core import BoundingBox, Document, Page, Point, TextSpan
from udml.core.models import build_source_groups
from udml.structure.reconstruction import reconstructed_tables
from udml.structure.relations import table_relationships
from udml.structure.tables import (
    RuledGridCandidate,
    RuledGridCell,
    RuledGridConfig,
    SpanCellAssignment,
)

_EXTRACTED_AT = datetime(
    2026, 10, 16, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
)


def _configuration() -> RuledGridConfig:
    return RuledGridConfig(
        axis_tolerance=0.5,
        snap_tolerance=0.5,
        gap_tolerance=1.0,
        containment_tolerance=1.0,
    )


def _span(span_index: int, text: str, x: float) -> TextSpan:
    return TextSpan(
        id=f"p1-b0-l0-s{span_index}",
        text=text,
        bbox=BoundingBox(x0=x, y0=2, x1=x + 6, y1=8),
        font_name="Roboto-Regular",
        font_size=5,
        font_flags=0,
        origin=Point(x=x, y=7),
        block_index=0,
        line_index=0,
        span_index=span_index,
    )


def _document(spans: list[TextSpan]) -> Document:
    lines, blocks = build_source_groups(1, spans, {(0, 0): ((1.0, 0.0), 0)})
    return Document(
        source_filename="grids.pdf",
        source_sha256="ab" * 32,
        parser_name="test",
        parser_version="0",
        extracted_at=_EXTRACTED_AT,
        pages=[
            Page(
                number=1,
                width=200,
                height=100,
                rotation=0,
                spans=spans,
                lines=lines,
                blocks=blocks,
            )
        ],
    )


def _assignment(span: TextSpan, cell: BoundingBox, method: str) -> SpanCellAssignment:
    return SpanCellAssignment(
        span_id=span.id,
        cell=cell,
        method=method,
        bbox=span.bbox,
        origin=span.origin,
    )


def _candidate(
    cells: list[tuple[BoundingBox, list[str]]],
    assignments: list[SpanCellAssignment],
) -> RuledGridCandidate:
    boxes = [box for box, _span_ids in cells]
    return RuledGridCandidate(
        id="p1-g0",
        page_number=1,
        bbox=BoundingBox(
            x0=min(box.x0 for box in boxes),
            y0=min(box.y0 for box in boxes),
            x1=max(box.x1 for box in boxes),
            y1=max(box.y1 for box in boxes),
        ),
        cells=[RuledGridCell(bbox=box, span_ids=span_ids) for box, span_ids in cells],
        unassigned_span_ids=[],
        assignments=assignments,
        configuration=_configuration(),
    )


def test_strict_and_fallback_assignments_appear_once() -> None:
    strict = _span(0, " 4.64 ", 2)
    fallback = _span(1, "23-Jun-2026", 12)
    cell = BoundingBox(x0=0, y0=0, x1=30, y1=10)
    candidate = _candidate(
        [(cell, [strict.id])],
        [
            _assignment(strict, cell, "bbox-containment"),
            _assignment(fallback, cell, "baseline-fallback"),
        ],
    )
    reconstructed = reconstructed_tables(_document([strict, fallback]), [candidate])

    assert len(reconstructed) == 1
    placed = reconstructed[0].cells[0]
    assert [(item.span_id, item.method, item.text) for item in placed.assignments] == [
        (strict.id, "bbox-containment", " 4.64 "),
        (fallback.id, "baseline-fallback", "23-Jun-2026"),
    ]
    assert placed.lines[0].order == "local-geometric"
    assert [item.text for item in placed.lines[0].spans] == [" 4.64 ", "23-Jun-2026"]
    assert fallback.id not in candidate.cells[0].span_ids


def test_missing_slot_stays_distinct_from_an_empty_supported_cell() -> None:
    occupied_box = BoundingBox(x0=10, y0=0, x1=20, y1=10)
    empty_box = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    lower_box = BoundingBox(x0=0, y0=10, x1=10, y1=20)
    span = _span(0, "4.64", 12)
    candidate = _candidate(
        [(empty_box, []), (occupied_box, [span.id]), (lower_box, [])],
        [_assignment(span, occupied_box, "bbox-containment")],
    )
    reconstructed = reconstructed_tables(_document([span]), [candidate])

    table = reconstructed[0]
    slots = {(cell.row, cell.column): cell for cell in table.cells}
    assert table.x_boundaries == [0, 10, 20]
    assert table.y_boundaries == [0, 10, 20]
    assert (1, 1) not in slots
    empty = slots[(0, 0)]
    assert empty.bbox == empty_box
    assert empty.assignments == []
    assert empty.lines == []
    assert [item.span_id for item in slots[(0, 1)].assignments] == [span.id]


def test_conflicting_cell_assignments_remain_unresolved() -> None:
    span = _span(0, "4.64", 2)
    left = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    right = BoundingBox(x0=10, y0=0, x1=20, y1=10)
    candidate = _candidate(
        [(left, [span.id]), (right, [])],
        [
            _assignment(span, left, "bbox-containment"),
            _assignment(span, right, "baseline-fallback"),
        ],
    )
    table = reconstructed_tables(_document([span]), [candidate])[0]

    assert all(
        span.id not in [item.span_id for item in cell.assignments]
        for cell in table.cells
    )
    membership = table.unresolved_memberships[0]
    assert membership.span_id == span.id
    assert [(item.method, item.cell) for item in membership.candidates] == [
        ("bbox-containment", left),
        ("baseline-fallback", right),
    ]
    contained = [
        item
        for item in table_relationships([table])
        if item.type == "cell-contains-span" and span.id in item.span_ids
    ]
    assert contained == []


def test_column_header_candidate_keeps_evidence_and_status() -> None:
    header = _span(0, "Name", 2)
    first = _span(1, "4.64", 2)
    second = _span(2, "1.20", 2)
    top = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    middle = BoundingBox(x0=0, y0=10, x1=10, y1=20)
    bottom = BoundingBox(x0=0, y0=20, x1=10, y1=30)
    candidate = _candidate(
        [(top, [header.id]), (middle, [first.id]), (bottom, [second.id])],
        [
            _assignment(header, top, "bbox-containment"),
            _assignment(first, middle, "bbox-containment"),
            _assignment(second, bottom, "bbox-containment"),
        ],
    )
    table = reconstructed_tables(_document([header, first, second]), [candidate])[0]
    links = [
        item for item in table_relationships([table]) if item.type == "column-header"
    ]

    assert len(links) == 1
    link = links[0]
    assert link.status == "candidate"
    assert link.method == "column-header-candidate-v1"
    assert link.span_ids == [header.id]
    assert link.reason is not None and "Titles and data" in link.reason
    placed = next(cell for cell in table.cells if cell.row == 0)
    assert [item.text for item in placed.assignments] == ["Name"]
