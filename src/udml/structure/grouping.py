"""Geometric neighbor candidates between source text blocks.

Below and right are directions in unrotated page space: x increases
rightward and y increases downward. A result is a candidate edge. It does
not merge blocks, order them for reading, label columns, or identify tables.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from udml.core.geometry import BoundingBox
from udml.core.models import Document, TextBlock
from udml.core.relationships import GEOMETRY_NEIGHBORS_V1, BlockNeighbor

_Relationship = Literal["below", "right"]


class NeighborConfig(BaseModel):
    """Thresholds for ``block_neighbor_candidates``.

    ``min_overlap_ratio`` is finite and in the closed range 0 to 1. It is the
    minimum allowed perpendicular overlap ratio.

    ``max_gap_fraction`` is finite and greater than or equal to 0. The
    maximum gap in PDF points is this fraction times the page height for
    "below", or times the page width for "right". There is no upper cap, so
    a value above 1 allows a gap larger than that page dimension.
    """

    model_config = ConfigDict(extra="forbid")

    min_overlap_ratio: Annotated[FiniteFloat, Field(ge=0, le=1)]
    max_gap_fraction: Annotated[FiniteFloat, Field(ge=0)]


def block_neighbor_candidates(
    document: Document, config: NeighborConfig
) -> list[BlockNeighbor]:
    """Return at most one below and one right candidate for each block.

    Only text blocks with positive area are sources or targets. Blocks on
    different pages are never compared. ``document`` is not modified.

    For a source box S and a target box T, the movement-axis rule is:

    - Below: T is eligible only when ``T.y0 >= S.y1``. The vertical intervals
      are then separated or touching, and vertical overlap is excluded. The
      gap is ``T.y0 - S.y1`` in PDF points.
    - Right: T is eligible only when ``T.x0 >= S.x1``. The gap is
      ``T.x0 - S.x1``.

    The gap must be less than or equal to ``max_gap_fraction`` times the
    page height (below) or the page width (right). Those dimensions are the
    unrotated values stored on the page.

    The perpendicular overlap must be positive. Below uses the horizontal
    intervals; right uses the vertical intervals. Intervals that only touch
    have overlap zero and are rejected. The overlap ratio is that positive
    length divided by the smaller box extent on the same axis, and it must
    be greater than or equal to ``min_overlap_ratio``.

    Among eligible targets, the winner has the smallest gap, then the
    highest overlap ratio, then the lexicographically smallest target id.
    Comparisons are exact. The returned list follows document page order
    and each page's block order, with below before right. That sequence is
    not a reading order.
    """
    candidates: list[BlockNeighbor] = []
    for page in document.pages:
        blocks = [block for block in page.blocks if _positive_area(block.bbox)]
        for source in blocks:
            for relationship in ("below", "right"):
                chosen = _nearest(
                    source,
                    blocks,
                    relationship,
                    page.height,
                    page.width,
                    config,
                )
                if chosen is None:
                    continue
                target, gap, ratio = chosen
                candidates.append(
                    BlockNeighbor(
                        page_number=page.number,
                        source_block_id=source.id,
                        target_block_id=target.id,
                        relationship_type=relationship,
                        gap=gap,
                        overlap_ratio=ratio,
                        method=GEOMETRY_NEIGHBORS_V1,
                    )
                )
    return candidates


def _positive_area(box: BoundingBox) -> bool:
    return box.width > 0 and box.height > 0


def _overlap_length(
    start_a: float, end_a: float, start_b: float, end_b: float
) -> float:
    """Overlap of two closed intervals. Touching or separation yields zero."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _nearest(
    source: TextBlock,
    blocks: list[TextBlock],
    relationship: _Relationship,
    page_height: float,
    page_width: float,
    config: NeighborConfig,
) -> tuple[TextBlock, float, float] | None:
    best: tuple[float, float, str, TextBlock, float] | None = None
    for target in blocks:
        if target.id == source.id:
            continue
        scored = _score(
            source.bbox,
            target.bbox,
            relationship,
            page_height,
            page_width,
            config,
        )
        if scored is None:
            continue
        gap, ratio = scored
        rank = (gap, -ratio, target.id)
        if best is None or rank < best[:3]:
            best = (gap, -ratio, target.id, target, ratio)
    if best is None:
        return None
    return best[3], best[0], best[4]


def _score(
    source: BoundingBox,
    target: BoundingBox,
    relationship: _Relationship,
    page_height: float,
    page_width: float,
    config: NeighborConfig,
) -> tuple[float, float] | None:
    if relationship == "below":
        if target.y0 < source.y1:
            return None
        gap = target.y0 - source.y1
        limit = config.max_gap_fraction * page_height
        overlap = _overlap_length(source.x0, source.x1, target.x0, target.x1)
        smaller = min(source.width, target.width)
    else:
        if target.x0 < source.x1:
            return None
        gap = target.x0 - source.x1
        limit = config.max_gap_fraction * page_width
        overlap = _overlap_length(source.y0, source.y1, target.y0, target.y1)
        smaller = min(source.height, target.height)
    if gap > limit or overlap <= 0 or smaller <= 0:
        return None
    ratio = overlap / smaller
    if ratio < config.min_overlap_ratio:
        return None
    return gap, ratio
