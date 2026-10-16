"""Ruled-grid candidates from stored vector segments.

A candidate is a set of elementary rectangles whose four sides are covered
by stroked horizontal and vertical segments. It is not a confirmed table,
a reading order, or a claim that every grid on the page is meaningful.

Limitations that this detector does not remove:

- Borderless tables have no stroked rulings to follow.
- Merged cells are not inferred, and missing sides are not invented.
- Dashed strokes stay split when a gap is larger than ``gap_tolerance``.
- Decorative rulings can produce candidates.
- Rotated text is not interpreted; only axis-aligned span boxes are used.
- A stroked line that crosses two grids joins them into one component.
- Loose tolerances can snap distinct rulings together.
- Stroke width is ignored, so hairline and heavy rules are treated alike.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from udml.core.geometry import BoundingBox, union_boxes
from udml.core.models import Document, Page

_Line = tuple[float, list[tuple[float, float]]]

RULED_GRID_V1: Literal["ruled-grid-v1"] = "ruled-grid-v1"

RULED_GRID_LIMITATIONS = (
    "These are ruled-grid candidates, not confirmed tables.",
    "Borderless tables are not detected.",
    "Merged cells are not inferred.",
    "Dashed strokes stay broken when the gap exceeds gap_tolerance.",
    "Decorative ruling can produce candidates.",
    "Rotated text is not handled; span boxes are axis-aligned.",
    "A ruling line that crosses two grids joins them into one component.",
    "Stroke width is ignored, so hairline and heavy rules are treated alike.",
)


class RuledGridConfig(BaseModel):
    """Tolerances, in PDF points, for ``ruled_grid_candidates``.

    Every value is finite and nonnegative. ``axis_tolerance`` is how far a
    stroked segment may depart from horizontal or vertical. ``snap_tolerance``
    is the largest spread of one coordinate cluster. ``gap_tolerance`` is the
    largest collinear opening that is still joined. ``containment_tolerance``
    is how far a span box may extend outside a cell and still count as inside.
    """

    model_config = ConfigDict(extra="forbid")

    axis_tolerance: Annotated[FiniteFloat, Field(ge=0)]
    snap_tolerance: Annotated[FiniteFloat, Field(ge=0)]
    gap_tolerance: Annotated[FiniteFloat, Field(ge=0)]
    containment_tolerance: Annotated[FiniteFloat, Field(ge=0)]


class RuledGridCell(BaseModel):
    """One elementary rectangle with all four ruled sides.

    ``span_ids`` are spans wholly inside this cell and no other cell of the
    same candidate. The order is lexicographic, not a reading order.
    """

    model_config = ConfigDict(extra="forbid")

    bbox: BoundingBox
    span_ids: list[str]


class RuledGridCandidate(BaseModel):
    """One page-local ruled-grid candidate.

    ``bbox`` is the union of the supported cells. It can cover a hole where
    no elementary cell was supported. ``cells`` follow increasing ``y0``,
    then increasing ``x0``, in unrotated page space. That is not a reading
    order. ``unassigned_span_ids`` are spans that meet the candidate region
    but do not lie wholly inside exactly one of its cells.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    page_number: Annotated[int, Field(ge=1)]
    bbox: BoundingBox
    cells: list[RuledGridCell] = Field(min_length=1)
    unassigned_span_ids: list[str]
    method: Literal["ruled-grid-v1"] = RULED_GRID_V1
    configuration: RuledGridConfig


def ruled_grid_candidates(
    document: Document, configuration: RuledGridConfig
) -> list[RuledGridCandidate]:
    """Return ruled-grid candidates for each page.

    Only stroked segments are used. Fill-only edges, zero-length segments,
    and diagonals are ignored. ``document`` is not modified. See
    ``RULED_GRID_LIMITATIONS``.

    Horizontal and vertical positions are clustered from the low coordinate
    upward or rightward. A later coordinate joins a cluster only while it
    remains within ``snap_tolerance`` of that cluster's first coordinate, so
    a chain cannot grow past the snapping tolerance. The snapped position is
    the median. Overlapping collinear intervals, and gaps no larger than
    ``gap_tolerance``, become one interval.

    An intersection joins a horizontal line and a vertical line when each
    covers the other's snapped coordinate within ``snap_tolerance``. A
    connected component is kept only when it has at least two horizontal and
    two vertical coordinates and at least one elementary cell. A cell exists
    only when each of its four sides is covered by a merged interval within
    ``snap_tolerance``. Adjacent coordinates are the only cell sizes; a
    larger rectangle is not substituted for missing internal rulings.
    """
    candidates: list[RuledGridCandidate] = []
    recorded = configuration.model_copy()
    for page in document.pages:
        candidates.extend(_page_candidates(page, recorded))
    return candidates


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        parent = self._parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


def _page_candidates(
    page: Page, configuration: RuledGridConfig
) -> list[RuledGridCandidate]:
    horizontals, verticals = _ruling_segments(page, configuration)
    if not horizontals or not verticals:
        return []
    slack = configuration.snap_tolerance
    groups = _components(horizontals, verticals, slack)
    cell_groups: list[list[BoundingBox]] = []
    for horizontal_ids, vertical_ids in groups:
        boxes = _cells(horizontals, verticals, horizontal_ids, vertical_ids, slack)
        if boxes:
            cell_groups.append(sorted(boxes, key=_box_key))
    cell_groups.sort(key=_geometry_key)
    regions = [union_boxes(boxes) for boxes in cell_groups]
    assigned, unassigned = _assign_spans(page, cell_groups, regions, configuration)
    candidates: list[RuledGridCandidate] = []
    for index, boxes in enumerate(cell_groups):
        cells = [
            RuledGridCell(bbox=box, span_ids=span_ids)
            for box, span_ids in zip(boxes, assigned[index], strict=True)
        ]
        candidates.append(
            RuledGridCandidate(
                id=f"p{page.number}-g{index}",
                page_number=page.number,
                bbox=regions[index],
                cells=cells,
                unassigned_span_ids=unassigned[index],
                method=RULED_GRID_V1,
                configuration=configuration.model_copy(),
            )
        )
    return candidates


def _ruling_segments(
    page: Page, configuration: RuledGridConfig
) -> tuple[list[_Line], list[_Line]]:
    horizontal: list[tuple[float, float, float]] = []
    vertical: list[tuple[float, float, float]] = []
    axis = configuration.axis_tolerance
    for segment in page.vector_segments:
        if not segment.stroked:
            continue
        x0 = segment.start.x
        y0 = segment.start.y
        x1 = segment.end.x
        y1 = segment.end.y
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx == 0 and dy == 0:
            continue
        if dy <= axis and dx > axis:
            horizontal.append(((y0 + y1) / 2, min(x0, x1), max(x0, x1)))
        elif dx <= axis and dy > axis:
            vertical.append(((x0 + x1) / 2, min(y0, y1), max(y0, y1)))
    return (
        _snap_and_merge(horizontal, configuration),
        _snap_and_merge(vertical, configuration),
    )


def _snap_and_merge(
    records: list[tuple[float, float, float]], configuration: RuledGridConfig
) -> list[_Line]:
    if not records:
        return []
    clusters = _clusters([record[0] for record in records], configuration.snap_tolerance)
    grouped: dict[float, list[tuple[float, float]]] = {}
    for position, start, end in records:
        snapped = _snap(position, clusters, configuration.snap_tolerance)
        grouped.setdefault(snapped, []).append((start, end))
    return [
        (position, _merge(grouped[position], configuration.gap_tolerance))
        for position in sorted(grouped)
    ]


def _clusters(values: list[float], tolerance: float) -> list[tuple[float, float]]:
    """Cluster sorted unique values without chaining past ``tolerance``.

    Each cluster starts at its lowest value. A value is included only when
    its distance from that start is at most ``tolerance``. The snapped
    coordinate is the median.
    """
    ordered = sorted(set(values))
    if not ordered:
        return []
    clusters: list[tuple[float, float]] = []
    anchor = ordered[0]
    members = [anchor]
    for value in ordered[1:]:
        if value - anchor <= tolerance:
            members.append(value)
            continue
        clusters.append((anchor, _median(members)))
        anchor = value
        members = [value]
    clusters.append((anchor, _median(members)))
    return clusters


def _median(values: list[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2


def _snap(value: float, clusters: list[tuple[float, float]], tolerance: float) -> float:
    for anchor, snapped in clusters:
        if anchor <= value <= anchor + tolerance:
            return snapped
    return value


def _merge(
    intervals: list[tuple[float, float]], gap_tolerance: float
) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _components(
    horizontals: list[_Line],
    verticals: list[_Line],
    slack: float,
) -> list[tuple[list[int], list[int]]]:
    horizontal_count = len(horizontals)
    union_find = _UnionFind(horizontal_count + len(verticals))
    for horizontal_index, (y_position, h_intervals) in enumerate(horizontals):
        for vertical_index, (x_position, v_intervals) in enumerate(verticals):
            crosses = _point_on(h_intervals, x_position, slack) and _point_on(
                v_intervals, y_position, slack
            )
            if crosses:
                union_find.union(horizontal_index, horizontal_count + vertical_index)
    grouped: dict[int, list[int]] = {}
    for index in range(horizontal_count + len(verticals)):
        grouped.setdefault(union_find.find(index), []).append(index)
    components: list[tuple[list[int], list[int]]] = []
    for indexes in grouped.values():
        horizontal_ids = [index for index in indexes if index < horizontal_count]
        vertical_ids = [
            index - horizontal_count
            for index in indexes
            if index >= horizontal_count
        ]
        if len(horizontal_ids) >= 2 and len(vertical_ids) >= 2:
            components.append((horizontal_ids, vertical_ids))
    return components


def _cells(
    horizontals: list[_Line],
    verticals: list[_Line],
    horizontal_ids: list[int],
    vertical_ids: list[int],
    slack: float,
) -> list[BoundingBox]:
    ys = sorted(horizontals[index][0] for index in horizontal_ids)
    xs = sorted(verticals[index][0] for index in vertical_ids)
    horizontal_intervals = {
        horizontals[index][0]: horizontals[index][1] for index in horizontal_ids
    }
    vertical_intervals = {
        verticals[index][0]: verticals[index][1] for index in vertical_ids
    }
    boxes: list[BoundingBox] = []
    for x_index in range(len(xs) - 1):
        x0, x1 = xs[x_index], xs[x_index + 1]
        for y_index in range(len(ys) - 1):
            y0, y1 = ys[y_index], ys[y_index + 1]
            covered = (
                _covers(horizontal_intervals[y0], x0, x1, slack)
                and _covers(horizontal_intervals[y1], x0, x1, slack)
                and _covers(vertical_intervals[x0], y0, y1, slack)
                and _covers(vertical_intervals[x1], y0, y1, slack)
            )
            if covered:
                boxes.append(BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))
    return boxes


def _point_on(
    intervals: list[tuple[float, float]], point: float, slack: float
) -> bool:
    return any(start - slack <= point <= end + slack for start, end in intervals)


def _covers(
    intervals: list[tuple[float, float]], start: float, end: float, slack: float
) -> bool:
    return any(left <= start + slack and right >= end - slack for left, right in intervals)


def _assign_spans(
    page: Page,
    cell_groups: list[list[BoundingBox]],
    regions: list[BoundingBox],
    configuration: RuledGridConfig,
) -> tuple[list[list[list[str]]], list[list[str]]]:
    """Assign a span only when it fits inside exactly one cell on the page."""
    tolerance = configuration.containment_tolerance
    assigned = [[[] for _ in boxes] for boxes in cell_groups]
    unassigned: list[list[str]] = [[] for _ in cell_groups]
    for span in page.spans:
        fits: list[tuple[int, int]] = []
        for candidate_index, boxes in enumerate(cell_groups):
            for cell_index, box in enumerate(boxes):
                if _contains(box, span.bbox, tolerance):
                    fits.append((candidate_index, cell_index))
        if len(fits) == 1:
            candidate_index, cell_index = fits[0]
            assigned[candidate_index][cell_index].append(span.id)
            continue
        for candidate_index, region in enumerate(regions):
            in_candidate = any(index == candidate_index for index, _ in fits)
            if in_candidate or _intersects(span.bbox, region):
                unassigned[candidate_index].append(span.id)
    for span_ids in assigned:
        for cell_ids in span_ids:
            cell_ids.sort()
    for span_ids in unassigned:
        span_ids.sort()
    return assigned, unassigned


def _contains(cell: BoundingBox, span: BoundingBox, tolerance: float) -> bool:
    return (
        span.x0 >= cell.x0 - tolerance
        and span.x1 <= cell.x1 + tolerance
        and span.y0 >= cell.y0 - tolerance
        and span.y1 <= cell.y1 + tolerance
    )


def _intersects(span: BoundingBox, region: BoundingBox) -> bool:
    return (
        span.x0 <= region.x1
        and span.x1 >= region.x0
        and span.y0 <= region.y1
        and span.y1 >= region.y0
    )


def _box_key(box: BoundingBox) -> tuple[float, float, float, float]:
    return (box.y0, box.x0, box.y1, box.x1)


def _geometry_key(
    boxes: list[BoundingBox],
) -> tuple[float, float, float, float, tuple[tuple[float, float, float, float], ...]]:
    ordered = sorted(boxes, key=_box_key)
    return (*_box_key(union_boxes(ordered)), tuple(_box_key(box) for box in ordered))
