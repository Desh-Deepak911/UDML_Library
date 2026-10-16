"""Page geometry for extracted PDF text.

Coordinates are PDF points: 72 points equal one inch. The origin is the
top-left of the unrotated page, with x increasing rightward and y increasing
downward. A box keeps the extracted coordinates when content falls outside
the page, including negative values. Rotation lives on the page, not in the
box.
"""

from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, FiniteFloat, model_validator


class Point(BaseModel):
    """A point in unrotated page space, in PDF points.

    The origin is the top-left of the unrotated page. ``x`` increases
    rightward and ``y`` increases downward.
    """

    model_config = ConfigDict(extra="forbid")

    x: FiniteFloat
    y: FiniteFloat


class BoundingBox(BaseModel):
    """Axis-aligned box in unrotated page space, in PDF points.

    ``x1`` is greater than or equal to ``x0``, and ``y1`` is greater than or
    equal to ``y0``. Zero width or height is valid.
    """

    model_config = ConfigDict(extra="forbid")

    x0: FiniteFloat
    y0: FiniteFloat
    x1: FiniteFloat
    y1: FiniteFloat

    @model_validator(mode="after")
    def corners_are_ordered(self) -> Self:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("x1 must be >= x0 and y1 must be >= y0")
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def union_boxes(boxes: Sequence[BoundingBox]) -> BoundingBox:
    """Return the union of ``boxes``.

    The result keeps coordinates that fall outside the page. It is not clamped.
    """
    if not boxes:
        raise ValueError("a box union needs at least one box")
    return BoundingBox(
        x0=min(box.x0 for box in boxes),
        y0=min(box.y0 for box in boxes),
        x1=max(box.x1 for box in boxes),
        y1=max(box.y1 for box in boxes),
    )
