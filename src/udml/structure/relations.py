"""Structural relationships derived from reconstructed tables.

The result is not stored on the document. Row and column links name the
group that contains a cell. They are not pairs among every cell in that
group. A column-header link is a candidate, not a semantic fact. Titles
and ordinary data can both satisfy that heuristic.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from udml.structure.reconstruction import (
    TABLE_RECONSTRUCTION_V1,
    TableCandidate,
    TableCell,
)

COLUMN_HEADER_CANDIDATE_V1: Literal["column-header-candidate-v1"] = (
    "column-header-candidate-v1"
)
_HEADER_REASON = (
    "Alphabetic text in the top supported band with two later cells "
    "in this column. Titles and data can both match."
)

_Kind = Literal["table", "cell", "span", "row", "column"]
_Type = Literal[
    "table-contains-cell",
    "cell-contains-span",
    "row-contains-cell",
    "column-contains-cell",
    "column-header",
]
_Method = Literal[
    "table-reconstruction-v1",
    "bbox-containment",
    "baseline-fallback",
    "column-header-candidate-v1",
]
_Status = Literal["structural", "candidate"]


class StructureRef(BaseModel):
    """One endpoint of a derived relationship."""

    model_config = ConfigDict(extra="forbid")

    kind: _Kind
    id: str


class StructuralRelationship(BaseModel):
    """One evidence-backed link between reconstructed structures.

    ``status`` is ``structural`` when reconstruction already established
    the link. ``candidate`` links include a short ``reason``. There is no
    confidence score. ``span_ids`` are source span ids, not normalized text.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: _Type
    source: StructureRef
    target: StructureRef
    method: _Method
    span_ids: list[str]
    status: _Status
    reason: str | None = None

    @model_validator(mode="after")
    def reason_matches_status(self) -> Self:
        if self.status == "candidate" and not self.reason:
            raise ValueError("a candidate relationship needs a reason")
        if self.status == "structural" and self.reason is not None:
            raise ValueError("a structural relationship has no reason")
        return self


def table_relationships(
    tables: list[TableCandidate],
) -> list[StructuralRelationship]:
    """Return relationships for ``tables`` without modifying them."""
    relationships: list[StructuralRelationship] = []
    for table in tables:
        relationships.extend(_table_relationships(table))
    return relationships


def _table_relationships(table: TableCandidate) -> list[StructuralRelationship]:
    cell_ids = [
        _cell_id(table.id, cell, index) for index, cell in enumerate(table.cells)
    ]
    relationships: list[StructuralRelationship] = []
    indexed: list[tuple[str, TableCell]] = []
    for cell_id, cell in zip(cell_ids, table.cells, strict=True):
        span_ids = [item.span_id for item in cell.assignments]
        relationships.append(
            _structural(
                f"{cell_id}-table",
                "table-contains-cell",
                StructureRef(kind="table", id=table.id),
                StructureRef(kind="cell", id=cell_id),
                TABLE_RECONSTRUCTION_V1,
                span_ids,
            )
        )
        for item in cell.assignments:
            relationships.append(
                _structural(
                    f"{cell_id}-span-{item.span_id}-{item.method}",
                    "cell-contains-span",
                    StructureRef(kind="cell", id=cell_id),
                    StructureRef(kind="span", id=item.span_id),
                    item.method,
                    [item.span_id],
                )
            )
        if cell.row is not None and cell.column is not None:
            indexed.append((cell_id, cell))
    relationships.extend(_groups(table.id, indexed))
    relationships.extend(_headers(table, indexed))
    return relationships


def _groups(
    table_id: str,
    indexed: list[tuple[str, TableCell]],
) -> list[StructuralRelationship]:
    relationships: list[StructuralRelationship] = []
    for cell_id, cell in sorted(indexed, key=lambda item: _slot_key(item[1])):
        span_ids = [item.span_id for item in cell.assignments]
        assert cell.row is not None and cell.column is not None
        relationships.append(
            _structural(
                f"{cell_id}-row",
                "row-contains-cell",
                StructureRef(kind="row", id=f"{table_id}-row{cell.row}"),
                StructureRef(kind="cell", id=cell_id),
                TABLE_RECONSTRUCTION_V1,
                span_ids,
            )
        )
        relationships.append(
            _structural(
                f"{cell_id}-col",
                "column-contains-cell",
                StructureRef(kind="column", id=f"{table_id}-col{cell.column}"),
                StructureRef(kind="cell", id=cell_id),
                TABLE_RECONSTRUCTION_V1,
                span_ids,
            )
        )
    return relationships


def _headers(
    table: TableCandidate,
    indexed: list[tuple[str, TableCell]],
) -> list[StructuralRelationship]:
    band = _top_band(indexed)
    rows_by_column: dict[int, set[int]] = {}
    for _cell_id, cell in indexed:
        assert cell.column is not None and cell.row is not None
        rows_by_column.setdefault(cell.column, set()).add(cell.row)
    relationships: list[StructuralRelationship] = []
    ordered = sorted(indexed, key=lambda item: _slot_key(item[1]))
    for cell_id, cell in ordered:
        assert cell.row is not None and cell.column is not None
        later_rows = sum(row > cell.row for row in rows_by_column[cell.column])
        if cell.row not in band or later_rows < 2 or not _has_alphabetic(cell):
            continue
        relationships.append(
            StructuralRelationship(
                id=f"{cell_id}-column-header",
                type="column-header",
                source=StructureRef(kind="cell", id=cell_id),
                target=StructureRef(kind="column", id=f"{table.id}-col{cell.column}"),
                method=COLUMN_HEADER_CANDIDATE_V1,
                span_ids=[item.span_id for item in cell.assignments],
                status="candidate",
                reason=_HEADER_REASON,
            )
        )
    return relationships


def _structural(
    relationship_id: str,
    relationship_type: _Type,
    source: StructureRef,
    target: StructureRef,
    method: _Method,
    span_ids: list[str],
) -> StructuralRelationship:
    return StructuralRelationship(
        id=relationship_id,
        type=relationship_type,
        source=source,
        target=target,
        method=method,
        span_ids=span_ids,
        status="structural",
    )


def _top_band(indexed: list[tuple[str, TableCell]]) -> set[int]:
    rows = sorted({cell.row for _cell_id, cell in indexed if cell.row is not None})
    band: set[int] = set()
    expected: int | None = None
    for row in rows:
        if expected is None:
            expected = row
        if row != expected:
            break
        band.add(row)
        expected = row + 1
    return band


def _has_alphabetic(cell: TableCell) -> bool:
    return any(
        item.text is not None and any(character.isalpha() for character in item.text)
        for item in cell.assignments
    )


def _cell_id(table_id: str, cell: TableCell, index: int) -> str:
    if cell.row is None or cell.column is None:
        return f"{table_id}-u{index}"
    return f"{table_id}-r{cell.row}c{cell.column}"


def _slot_key(cell: TableCell) -> tuple[int, int]:
    assert cell.row is not None and cell.column is not None
    return (cell.row, cell.column)
