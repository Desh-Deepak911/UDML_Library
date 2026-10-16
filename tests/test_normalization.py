from datetime import datetime, timedelta, timezone
from decimal import Decimal

from udml.core import BoundingBox, Document, Page, Point
from udml.core.geometry import union_boxes
from udml.core.models import TextBlock, TextLine, TextSpan
from udml.semantic import ContextClaim, SuppliedPair, normalize_observations
from udml.structure.reconstruction import (
    TableCandidate,
    TableCell,
    TableCellLine,
    TableSpanAssignment,
)
from udml.structure.relations import (
    StructuralRelationship,
    StructureRef,
)
from udml.structure.tables import RuledGridConfig

_EXTRACTED_AT = datetime(
    2026, 10, 16, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
)


def _document() -> Document:
    return Document(
        source_filename="sample.pdf",
        source_sha256="ab" * 32,
        parser_name="test",
        parser_version="0",
        extracted_at=_EXTRACTED_AT,
        pages=[Page(number=1, width=200, height=100, rotation=0)],
    )


def _configuration() -> RuledGridConfig:
    return RuledGridConfig(
        axis_tolerance=0.5,
        snap_tolerance=0.5,
        gap_tolerance=1.0,
        containment_tolerance=1.0,
    )


def _cell(row: int, column: int, text: str, span_id: str) -> TableCell:
    return TableCell(
        row=row,
        column=column,
        bbox=BoundingBox(x0=0, y0=row * 10, x1=10, y1=row * 10 + 10),
        assignments=[
            TableSpanAssignment(
                span_id=span_id,
                text=text,
                method="bbox-containment",
            )
        ],
        lines=[],
    )


def test_equivalent_labels_keep_wording_and_share_a_concept() -> None:
    observations = normalize_observations(
        _document(),
        pairs=[
            SuppliedPair(label_text="Net Asset Value", value_text="₹12.50"),
            SuppliedPair(label_text="NAV", value_text="Rs. 12.50"),
        ],
    )

    assert [item.concept_id for item in observations] == [
        "net_asset_value",
        "net_asset_value",
    ]
    assert [item.original_label for item in observations] == [
        "Net Asset Value",
        "NAV",
    ]
    assert observations[0].amount == Decimal("12.50")
    assert observations[1].denomination == "INR"
    assert observations[0].unit == "currency_per_unit"
    assert observations[1].unit == "currency_per_unit"
    assert observations[0].status == "resolved"
    assert observations[1].status == "resolved"
    assert observations[0].unit_evidence[-1].method == "concept-definition-v1"
    assert observations[0].unit_evidence[-1].source_text == "nav-currency-per-unit-v1"
    assert observations[0].source_sha256 == "ab" * 32


def test_percent_value_and_qualifiers_keep_their_evidence() -> None:
    observation = normalize_observations(
        _document(),
        pairs=[
            SuppliedPair(label_text="Total Expense Ratio", value_text="5.15%"),
        ],
        context={"plan": "direct", "effective_date": "2026-04-01"},
    )[0]

    assert observation.status == "resolved"
    assert observation.concept_id == "expense_ratio"
    assert observation.original_label == "Total Expense Ratio"
    assert observation.original_value == "5.15%"
    assert observation.amount == Decimal("5.15")
    assert observation.unit == "percent"
    assert observation.scale is None
    plan = next(item for item in observation.qualifiers if item.id == "plan")
    dated = next(item for item in observation.qualifiers if item.id == "effective_date")
    assert plan.value == "direct"
    assert plan.evidence[0].origin == "context"
    assert plan.evidence[0].source_text == "direct"
    assert dated.value == "2026-04-01"
    assert dated.evidence[0].origin == "context"
    assert dated.evidence[0].source_text == "2026-04-01"
    assert observation.entity_scope is None
    assert observation.unit_evidence[0].origin == "value"
    assert "%" in observation.unit_evidence[0].source_text


def test_ambiguous_missing_and_conflicting_inputs_are_not_resolved() -> None:
    document = _document()
    ambiguous = normalize_observations(
        document,
        pairs=[SuppliedPair(label_text="Net Assets", value_text="10")],
    )[0]
    assert ambiguous.status == "unresolved"
    assert ambiguous.concept_id is None
    assert ambiguous.concept_candidates == [
        "assets_under_management",
        "net_asset_value",
    ]

    missing = normalize_observations(
        document,
        pairs=[SuppliedPair(label_text="Expense Ratio", value_text="-")],
    )[0]
    assert missing.status == "unresolved"
    assert missing.amount is None

    conflict = normalize_observations(
        document,
        pairs=[SuppliedPair(label_text="direct plan nav", value_text="10%")],
        context={"plan": "regular"},
    )[0]
    assert conflict.status == "unresolved"

    header = _cell(0, 0, "Expense Ratio", "p1-b0-l0-s0")
    clean = _cell(1, 0, "5.15%", "p1-b0-l0-s1")
    disputed = _cell(2, 0, "4.00%", "p1-b0-l0-s2")
    table = TableCandidate(
        id="p1-g0",
        page_number=1,
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=30),
        cells=[header, clean, disputed],
        unassigned_span_ids=["p1-b0-l0-s2"],
        unresolved_memberships=[],
        configuration=_configuration(),
        x_boundaries=[0.0, 10.0],
        y_boundaries=[0.0, 10.0, 20.0, 30.0],
        warnings=[],
    )
    relationship = StructuralRelationship(
        id="p1-g0-r0c0-column-header",
        type="column-header",
        source=StructureRef(kind="cell", id="p1-g0-r0c0"),
        target=StructureRef(kind="column", id="p1-g0-col0"),
        method="column-header-candidate-v1",
        span_ids=["p1-b0-l0-s0"],
        status="candidate",
        reason="Header wording is only a candidate.",
    )
    paired = normalize_observations(
        document,
        [table],
        [relationship],
        context={"plan": "direct", "effective_date": "2026-04-01"},
    )
    by_value = {item.original_value: item for item in paired}
    assert by_value["5.15%"].status == "candidate"
    assert by_value["5.15%"].concept_id == "expense_ratio"
    assert by_value["5.15%"].amount == Decimal("5.15")
    assert by_value["5.15%"].pairing_method == "column-header-below-v1"
    assert by_value["4.00%"].status == "unresolved"
    assert "assigned-unassigned" in by_value["4.00%"].reasons
    assert all(item.status != "resolved" for item in paired)


def _span(
    span_id: str, text: str, x: float, line_index: int, span_index: int
) -> TextSpan:
    y0 = line_index * 12
    return TextSpan(
        id=span_id,
        text=text,
        bbox=BoundingBox(x0=x, y0=y0, x1=x + 8, y1=y0 + 8),
        font_name="test",
        font_size=5,
        font_flags=0,
        origin=Point(x=x, y=y0 + 6),
        block_index=0,
        line_index=line_index,
        span_index=span_index,
    )


def _lined_document(groups: list[list[TextSpan]]) -> Document:
    spans: list[TextSpan] = []
    lines: list[TextLine] = []
    for group in groups:
        spans.extend(group)
        lines.append(
            TextLine(
                id=f"p1-b0-l{group[0].line_index}",
                source_block_index=0,
                source_line_index=group[0].line_index,
                direction=(1, 0),
                writing_mode=0,
                span_ids=[span.id for span in group],
                bbox=union_boxes([span.bbox for span in group]),
            )
        )
    block = TextBlock(
        id="p1-b0",
        source_block_index=0,
        line_ids=[line.id for line in lines],
        bbox=union_boxes([line.bbox for line in lines]),
    )
    return _document().model_copy(
        update={
            "pages": [
                Page(
                    number=1,
                    width=200,
                    height=100,
                    rotation=0,
                    spans=spans,
                    lines=lines,
                    blocks=[block],
                )
            ]
        }
    )


def test_unit_scope_and_membership_conflicts_stay_unresolved() -> None:
    document = _document()
    unit_conflict = normalize_observations(
        document,
        pairs=[SuppliedPair(label_text="Expense Ratio", value_text="5.15%")],
        context={
            "plan": "direct",
            "effective_date": "2026-04-01",
            "unit": "INR",
        },
    )[0]
    assert unit_conflict.status == "unresolved"
    assert "conflicting-unit" in unit_conflict.reasons
    assert unit_conflict.denomination == "INR"
    assert unit_conflict.concept_id == "expense_ratio"
    plan = next(item for item in unit_conflict.qualifiers if item.id == "plan")
    assert plan.evidence[0].origin == "context"
    assert plan.evidence[0].source_text == "direct"

    scope_conflict = normalize_observations(
        document,
        pairs=[SuppliedPair(label_text="Expense Ratio", value_text="5.15%")],
        context={
            "plan": "direct",
            "effective_date": "2026-04-01",
            "entity_scope": "fund",
        },
    )[0]
    assert scope_conflict.status == "unresolved"
    assert "conflicting-scope" in scope_conflict.reasons
    assert scope_conflict.entity_scope is None
    assert scope_conflict.scope_evidence[0].source_text == "fund"
    assert scope_conflict.scope_evidence[0].origin == "context"

    disputed = _cell(1, 0, "4.00%", "p1-b0-l0-s1")
    table = TableCandidate(
        id="p1-g1",
        page_number=1,
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=20),
        cells=[_cell(0, 0, "Expense Ratio", "p1-b0-l0-s0"), disputed],
        unassigned_span_ids=["p1-b0-l0-s1"],
        unresolved_memberships=[],
        configuration=_configuration(),
        x_boundaries=[0.0, 10.0],
        y_boundaries=[0.0, 10.0, 20.0],
        warnings=[],
    )
    relationship = StructuralRelationship(
        id="p1-g1-r0c0-column-header",
        type="column-header",
        source=StructureRef(kind="cell", id="p1-g1-r0c0"),
        target=StructureRef(kind="column", id="p1-g1-col0"),
        method="column-header-candidate-v1",
        span_ids=["p1-b0-l0-s0"],
        status="candidate",
        reason="Header wording is only a candidate.",
    )
    paired = normalize_observations(
        document,
        [table],
        [relationship],
        context={"plan": "direct", "effective_date": "2026-04-01"},
    )[0]
    assert paired.status == "unresolved"
    assert "assigned-unassigned" in paired.reasons
    assert paired.concept_id == "expense_ratio"


def test_currency_fragments_and_attached_scale_stay_separate() -> None:
    mark = _span("p1-b0-l0-s0", "₹", 10, 0, 0)
    amount = _span("p1-b0-l0-s1", "38.75", 30, 0, 1)
    other = _span("p1-b0-l1-s0", "98.17", 10, 1, 0)
    document = _lined_document([[mark, amount], [other]])
    nav = normalize_observations(
        document,
        pairs=[
            SuppliedPair(
                label_text="NAV",
                value_span_ids=[amount.id, mark.id],
                page_number=1,
            )
        ],
    )[0]
    assert nav.original_value == "₹ 38.75"
    assert nav.value_span_ids == [mark.id, amount.id]
    assert nav.amount == Decimal("38.75")
    assert nav.denomination == "INR"
    assert nav.unit == "currency_per_unit"
    assert nav.scale is None
    assert nav.concept_id == "net_asset_value"
    assert nav.status == "resolved"
    assert other.id not in nav.value_span_ids

    both = normalize_observations(
        document,
        pairs=[
            SuppliedPair(
                label_text="NAV",
                value_span_ids=[amount.id, other.id],
            )
        ],
    )[0]
    assert both.amount is None
    assert "multiple-values" in both.reasons
    assert both.status == "unresolved"

    managed = normalize_observations(
        _document(),
        pairs=[
            SuppliedPair(
                label_text="Assets under management",
                value_text="₹108,546.98Cr.",
            )
        ],
    )[0]
    assert managed.amount == Decimal("108546.98")
    assert managed.denomination == "INR"
    assert managed.scale == "crore"
    assert managed.unit == "currency"
    assert managed.concept_id == "assets_under_management"
    assert managed.status == "unresolved"
    assert "missing-context" in managed.reasons
    assert "aum_basis" in managed.missing_context
    assert "Cr" in managed.scale_evidence[0].source_text

    ratio = normalize_observations(
        _document(),
        pairs=[SuppliedPair(label_text="Expense Ratio", value_text="Regular: 1.27%")],
    )[0]
    assert ratio.amount == Decimal("1.27")
    assert ratio.unit == "percent"
    assert ratio.qualifiers[0].id == "plan"
    assert ratio.qualifiers[0].value == "regular"
    assert ratio.qualifiers[0].evidence[0].source_text == "Regular:"
    assert ratio.status == "resolved"
    assert "effective_date" in ratio.missing_context
    assert "missing-context" not in ratio.reasons
    assert "plan" not in ratio.missing_context

    doubled = normalize_observations(
        _document(),
        pairs=[SuppliedPair(label_text="Expense Ratio", value_text="1.82%/2.47%")],
    )[0]
    assert doubled.amount is None
    assert "multiple-values" in doubled.reasons
    assert doubled.status == "unresolved"


def test_multiline_label_keeps_candidate_status() -> None:
    bare = normalize_observations(
        _document(),
        pairs=[SuppliedPair(label_text="Net Assets", value_text="4.25%")],
    )[0]
    assert bare.status == "unresolved"
    assert bare.concept_id is None
    assert "portfolio_weight" not in bare.concept_candidates

    percent = _span("p1-b0-l0-s0", "% to", 10, 0, 0)
    assets = _span("p1-b0-l1-s0", "Net Assets", 10, 1, 0)
    document = _lined_document([[percent], [assets]])
    joined = normalize_observations(
        document,
        pairs=[
            SuppliedPair(
                label_line_ids=["p1-b0-l0", "p1-b0-l1"],
                value_text="4.25%",
            )
        ],
    )[0]
    assert joined.original_label == "% to Net Assets"
    assert "|" not in joined.original_label
    assert joined.label_span_ids == [percent.id, assets.id]
    assert joined.concept_id is None
    assert joined.concept_candidates == ["portfolio_weight"]
    assert joined.status == "unresolved"

    first = TableSpanAssignment(
        span_id="p1-b0-l0-s0", text="% to", method="bbox-containment"
    )
    second = TableSpanAssignment(
        span_id="p1-b0-l1-s0", text="Net Assets", method="bbox-containment"
    )
    header = TableCell(
        row=0,
        column=0,
        bbox=BoundingBox(x0=0, y0=0, x1=20, y1=10),
        assignments=[first, second],
        lines=[
            TableCellLine(
                line_id="p1-b0-l0",
                direction=(1, 0),
                writing_mode=0,
                order="local-geometric",
                spans=[first],
            ),
            TableCellLine(
                line_id="p1-b0-l1",
                direction=(1, 0),
                writing_mode=0,
                order="local-geometric",
                spans=[second],
            ),
        ],
    )
    table = TableCandidate(
        id="p1-g2",
        page_number=1,
        bbox=BoundingBox(x0=0, y0=0, x1=20, y1=20),
        cells=[header, _cell(1, 0, "4.25%", "p1-b0-l2-s0")],
        unassigned_span_ids=[],
        unresolved_memberships=[],
        configuration=_configuration(),
        x_boundaries=[0.0, 20.0],
        y_boundaries=[0.0, 10.0, 20.0],
        warnings=[],
    )
    relationship = StructuralRelationship(
        id="p1-g2-r0c0-column-header",
        type="column-header",
        source=StructureRef(kind="cell", id="p1-g2-r0c0"),
        target=StructureRef(kind="column", id="p1-g2-col0"),
        method="column-header-candidate-v1",
        span_ids=["p1-b0-l0-s0", "p1-b0-l1-s0"],
        status="candidate",
        reason="Header wording is only a candidate.",
    )
    paired = normalize_observations(
        _document(),
        [table],
        [relationship],
        context={"subject": "holding"},
    )[0]
    assert paired.original_label == "% to Net Assets"
    assert paired.status == "candidate"
    assert paired.concept_id == "portfolio_weight"
    assert paired.amount == Decimal("4.25")
    assert paired.unit == "percent"
    assert paired.entity_scope == "holding"
    assert paired.pairing_method == "column-header-below-v1"


def test_plan_type_and_idcw_option_do_not_conflict() -> None:
    row = _span("p1-b0-l0-s0", "Direct Plan IDCW Option", 10, 0, 0)
    document = _lined_document([[row]])
    observation = normalize_observations(
        document,
        pairs=[SuppliedPair(label_text="NAV", value_text="₹38.75", page_number=1)],
        claims=[
            ContextClaim(
                id="plan",
                interpretation="direct",
                page_number=1,
                span_ids=[row.id],
            ),
            ContextClaim(
                id="option",
                interpretation="idcw",
                page_number=1,
                span_ids=[row.id],
            ),
        ],
    )[0]
    assert observation.status == "resolved"
    assert observation.unit == "currency_per_unit"
    assert observation.denomination == "INR"
    assert "conflicting-scope" not in observation.reasons
    option = next(item for item in observation.qualifiers if item.id == "option")
    assert option.value == "idcw"
    assert option.evidence[0].source_text == "Direct Plan IDCW Option"
    assert option.evidence[0].method == "caller-source-v1"
    assert option.evidence[0].span_ids == [row.id]
    assert option.evidence[0].document_sha256 == document.source_sha256
    assert option.evidence[0].page_number == 1


def test_named_rules_separate_inference_from_source_text() -> None:
    nav = normalize_observations(
        _document(),
        pairs=[SuppliedPair(label_text="NAV", value_text="₹12.50")],
    )[0]
    inferred = nav.unit_evidence[-1]
    assert inferred.method == "concept-definition-v1"
    assert inferred.source_text == "nav-currency-per-unit-v1"
    assert nav.denomination_evidence[0].source_text == "₹"
    assert nav.status == "resolved"

    as_on = _span("p1-b0-l0-s0", "As on September 30, 2026", 10, 0, 0)
    average = _span("p1-b0-l1-s0", "Average for Month of September, 2026", 10, 1, 0)
    document = _lined_document([[as_on], [average]])
    pointed = normalize_observations(
        document,
        pairs=[
            SuppliedPair(
                label_text="Assets under management",
                value_text="₹108,546.98Cr.",
                page_number=1,
            )
        ],
        claims=[
            ContextClaim(
                id="as_of_date",
                interpretation="2026-09-30",
                page_number=1,
                span_ids=[as_on.id],
            )
        ],
    )[0]
    basis = next(item for item in pointed.qualifiers if item.id == "aum_basis")
    assert basis.value == "point_in_time"
    assert basis.evidence[0].method == "concept-definition-v1"
    assert basis.evidence[0].source_text == "aum-as-on-point-in-time-v1"
    assert basis.evidence[0].span_ids == [as_on.id]
    assert pointed.amount == Decimal("108546.98")
    assert pointed.scale == "crore"
    assert pointed.status == "resolved"

    averaged = normalize_observations(
        document,
        pairs=[
            SuppliedPair(
                label_text="Assets under management",
                value_text="₹111,285.67Cr.",
                page_number=1,
            )
        ],
        claims=[
            ContextClaim(
                id="as_of_date",
                interpretation="2026-09-30",
                page_number=1,
                span_ids=[as_on.id, average.id],
            )
        ],
    )[0]
    assert averaged.status == "unresolved"
    assert "conflicting-basis" in averaged.reasons
    assert all(item.id != "aum_basis" for item in averaged.qualifiers)


def test_as_of_date_does_not_invent_an_effective_date() -> None:
    note = _span("p1-b0-l0-s0", "(As On September 30, 2026)", 10, 0, 0)
    document = _lined_document([[note]])
    observation = normalize_observations(
        document,
        pairs=[
            SuppliedPair(
                label_text="Expense Ratio",
                value_text="Regular: 1.27%",
                page_number=1,
            )
        ],
        claims=[
            ContextClaim(
                id="as_of_date",
                interpretation="2026-09-30",
                page_number=1,
                span_ids=[note.id],
            )
        ],
    )[0]
    assert observation.status == "resolved"
    dated = next(item for item in observation.qualifiers if item.id == "as_of_date")
    assert dated.value == "2026-09-30"
    assert dated.evidence[0].source_text == "(As On September 30, 2026)"
    assert dated.evidence[0].method == "caller-source-v1"
    assert "effective_date" in observation.missing_context
    assert all(item.id != "effective_date" for item in observation.qualifiers)

    rejected = normalize_observations(
        document,
        pairs=[SuppliedPair(label_text="Expense Ratio", value_text="Regular: 1.27%")],
        claims=[
            ContextClaim(
                id="as_of_date",
                interpretation="2026-09-30",
                page_number=1,
                span_ids=["missing-span"],
            )
        ],
    )[0]
    assert rejected.status == "unresolved"
    assert "missing-span" in rejected.reasons
    assert all(item.id != "as_of_date" for item in rejected.qualifiers)
