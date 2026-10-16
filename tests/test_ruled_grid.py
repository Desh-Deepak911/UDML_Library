from datetime import datetime, timedelta, timezone

from udml.core import BoundingBox, Document, Page, Point, TextSpan, VectorSegment
from udml.core.models import build_source_groups
from udml.structure.tables import (
    BASELINE_FALLBACK,
    RuledGridConfig,
    ruled_grid_candidates,
)

_EXTRACTED_AT = datetime(
    2026, 10, 16, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
)


def _segment(index: int, x0: float, y0: float, x1: float, y1: float) -> VectorSegment:
    return VectorSegment(
        id=f"p1-d0-i{index}-e0",
        start=Point(x=x0, y=y0),
        end=Point(x=x1, y=y1),
        source_drawing_index=0,
        source_item_index=index,
        source_edge_index=0,
        stroked=True,
        filled=False,
    )


def test_shared_horizontal_coordinate_does_not_bridge_a_gutter() -> None:
    segments = [
        _segment(0, 10, 10, 50, 10),
        _segment(1, 10, 40, 50, 40),
        _segment(2, 10, 10, 10, 40),
        _segment(3, 50, 10, 50, 40),
        _segment(4, 90, 10, 130, 10),
        _segment(5, 90, 40, 130, 40),
        _segment(6, 90, 10, 90, 40),
        _segment(7, 130, 10, 130, 40),
    ]
    document = Document(
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
                vector_segments=segments,
            )
        ],
    )
    configuration = RuledGridConfig(
        axis_tolerance=0.5,
        snap_tolerance=0.5,
        gap_tolerance=1.0,
        containment_tolerance=1.0,
    )
    candidates = ruled_grid_candidates(document, configuration)

    assert len(candidates) == 2
    assert [len(candidate.cells) for candidate in candidates] == [1, 1]
    left, right = candidates
    assert (left.bbox.x0, left.bbox.y0, left.bbox.x1, left.bbox.y1) == (10, 10, 50, 40)
    assert (right.bbox.x0, right.bbox.y0, right.bbox.x1, right.bbox.y1) == (
        90,
        10,
        130,
        40,
    )
    assert left.cells[0].bbox == left.bbox
    assert right.cells[0].bbox == right.bbox
    assert all(
        candidate.bbox.x1 <= 50 or candidate.bbox.x0 >= 90 for candidate in candidates
    )


def _configuration() -> RuledGridConfig:
    return RuledGridConfig(
        axis_tolerance=0.5,
        snap_tolerance=0.5,
        gap_tolerance=1.0,
        containment_tolerance=1.0,
    )


def _document(segments: list[VectorSegment], span: TextSpan) -> Document:
    lines, blocks = build_source_groups(1, [span], {(0, 0): ((1.0, 0.0), 0)})
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
                spans=[span],
                lines=lines,
                blocks=blocks,
                vector_segments=segments,
            )
        ],
    )


def _span(y0: float, y1: float, origin_y: float) -> TextSpan:
    return TextSpan(
        id="p1-b0-l0-s0",
        text="4.64",
        bbox=BoundingBox(x0=15, y0=y0, x1=40, y1=y1),
        font_name="Roboto-Regular",
        font_size=5,
        font_flags=0,
        origin=Point(x=15, y=origin_y),
        block_index=0,
        line_index=0,
        span_index=0,
    )


def test_interior_baseline_assigns_a_box_that_crosses_the_cell_top() -> None:
    segments = [
        _segment(0, 10, 10, 50, 10),
        _segment(1, 10, 30, 50, 30),
        _segment(2, 10, 10, 10, 30),
        _segment(3, 50, 10, 50, 30),
    ]
    span = _span(y0=8, y1=28, origin_y=20)
    document = _document(segments, span)
    configuration = _configuration()

    strict = ruled_grid_candidates(document, configuration)
    assert strict[0].cells[0].span_ids == []
    assert strict[0].assignments == []
    assert strict[0].unassigned_span_ids == [span.id]

    candidates = ruled_grid_candidates(
        document, configuration, baseline_fallback=True
    )
    candidate = candidates[0]
    assert candidate.cells[0].span_ids == []
    assert candidate.unassigned_span_ids == []
    assert len(candidate.assignments) == 1
    record = candidate.assignments[0]
    assert record.span_id == span.id
    assert record.method == BASELINE_FALLBACK
    assert record.cell == candidate.cells[0].bbox
    assert record.bbox == span.bbox
    assert record.origin == span.origin


def test_baseline_on_a_shared_boundary_stays_unassigned() -> None:
    segments = [
        _segment(0, 10, 10, 50, 10),
        _segment(1, 10, 30, 50, 30),
        _segment(2, 10, 50, 50, 50),
        _segment(3, 10, 10, 10, 50),
        _segment(4, 50, 10, 50, 50),
    ]
    span = _span(y0=26, y1=34, origin_y=30)
    candidates = ruled_grid_candidates(
        _document(segments, span),
        _configuration(),
        baseline_fallback=True,
    )
    candidate = candidates[0]
    assert len(candidate.cells) == 2
    assert all(cell.span_ids == [] for cell in candidate.cells)
    assert candidate.assignments == []
    assert candidate.unassigned_span_ids == [span.id]
