"""Geometric relationships derived from an extracted document.

These values are candidates computed from block boxes. They are not stored
on the document and do not assert paragraphs, tables, columns, or reading
order.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

GEOMETRY_NEIGHBORS_V1: Literal["geometry-neighbors-v1"] = "geometry-neighbors-v1"


class BlockNeighbor(BaseModel):
    """One directed geometric candidate between two source text blocks.

    ``gap`` is the separation on the movement axis in PDF points. Zero means
    the boxes touch on that axis. ``overlap_ratio`` is the perpendicular
    overlap length divided by the smaller box extent on that axis.
    ``method`` identifies the rule that produced the candidate.
    """

    model_config = ConfigDict(extra="forbid")

    page_number: Annotated[int, Field(ge=1)]
    source_block_id: str
    target_block_id: str
    relationship_type: Literal["below", "right"]
    gap: Annotated[FiniteFloat, Field(ge=0)]
    overlap_ratio: Annotated[FiniteFloat, Field(gt=0)]
    method: Literal["geometry-neighbors-v1"] = GEOMETRY_NEIGHBORS_V1

    @model_validator(mode="after")
    def endpoints_differ(self) -> Self:
        if self.source_block_id == self.target_block_id:
            raise ValueError("a block cannot neighbor itself")
        return self
