"""Derived observations from labels, values, and reconstructed tables.

A resolved observation means these evidence rules passed. It does not mean
the figure was checked against an external financial source. Currency and
scale are not converted. Nothing here is written back onto the document.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from udml.core.models import Document, TextLine, TextSpan
from udml.semantic.resolve import LABEL_RESOLUTION_V1, resolve_label
from udml.semantic.vocabulary import Vocabulary, initial_vocabulary, normalize_label
from udml.structure.reconstruction import TableCandidate, TableCell
from udml.structure.relations import StructuralRelationship

PAIR_CALLER_V1: Literal["caller-supplied-pair-v1"] = "caller-supplied-pair-v1"
PAIR_HEADER_V1: Literal["column-header-below-v1"] = "column-header-below-v1"

_Origin = Literal["label", "value", "context", "alias", "concept-definition"]
_Method = Literal[
    "source-text-v1",
    "caller-context-v1",
    "caller-source-v1",
    "concept-definition-v1",
]
_Status = Literal["candidate", "resolved", "unresolved"]
_MARKER_KEYS = frozenset({"unit", "scale", "entity_scope"})
_HORIZONTAL_DIRECTION_TOLERANCE = 1e-3
_PHRASES = (
    ("direct plan", "plan", "direct"),
    ("regular plan", "plan", "regular"),
    ("growth option", "option", "growth"),
    ("dividend option", "option", "dividend"),
    ("idcw option", "option", "idcw"),
    ("point in time", "aum_basis", "point_in_time"),
)
_RANGE = re.compile(r"(?i)\d\s*(?:to|through|[-–—])\s*\d")
_INEQUALITY = re.compile(r"(<=|>=|<|>|≤|≥)")
_AND = re.compile(r"(?i)\d\s+and\s+\d")
_PERCENT_SUFFIX = re.compile(
    r"(?i)\s*(?:%|(?<!\w)percent(?!\w)|(?<!\w)per\s+cent(?!\w))\s*$"
)
_PERCENT_WORD = re.compile(r"(?i)(?<!\w)percent(?!\w)|(?<!\w)per\s+cent(?!\w)")
_CURRENCY = (
    (re.compile("₹"), "INR"),
    (re.compile(r"(?i)(?<!\w)inr(?!\w)"), "INR"),
    (re.compile(r"(?i)(?<!\w)rs\.?(?!\w)"), "INR"),
)
_SCALE = (
    (re.compile(r"(?i)(?<![A-Za-z])crores?(?!\w)"), "crore"),
    (re.compile(r"(?i)(?<![A-Za-z])crs?\.?(?!\w)"), "crore"),
    (re.compile(r"(?i)(?<![A-Za-z])lakhs?(?!\w)"), "lakh"),
    (re.compile(r"(?i)(?<![A-Za-z])lacs?(?!\w)"), "lakh"),
)
_QUALIFIER_PREFIX = re.compile(r"(?i)^\s*(regular|direct)\s*:\s*")
_PREFIX_VALUES = {"regular": ("plan", "regular"), "direct": ("plan", "direct")}


class SourcedValue(BaseModel):
    """One established token and the text it came from."""

    model_config = ConfigDict(extra="forbid")

    value: str
    origin: _Origin
    source_text: str
    document_sha256: str | None = None
    page_number: int | None = None
    span_ids: list[str] = Field(default_factory=list)
    interpretation: str | None = None
    method: _Method | None = None


class ObservationQualifier(BaseModel):
    """A qualifier value with every piece of evidence that stated it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    value: str
    evidence: list[SourcedValue]


class NormalizedObservation(BaseModel):
    """One derived label and value.

    ``status`` is ``resolved`` only when the evidence rules in this module
    passed. That is not an external check of financial truth. ``amount`` uses
    ``Decimal`` and is never produced from a binary float. ``unit`` is a
    vocabulary unit kind such as ``currency`` or ``percent``. ``denomination``
    is a currency code such as ``INR`` and does not by itself select that kind.
    """

    model_config = ConfigDict(extra="forbid")

    concept_id: str | None
    concept_candidates: list[str]
    vocabulary_version: str
    original_label: str
    original_value: str
    amount: Decimal | None
    unit: str | None
    denomination: str | None
    scale: str | None
    unit_evidence: list[SourcedValue]
    denomination_evidence: list[SourcedValue]
    scale_evidence: list[SourcedValue]
    entity_scope: str | None
    scope_evidence: list[SourcedValue]
    qualifiers: list[ObservationQualifier]
    source_sha256: str
    page_number: int | None
    table_id: str | None
    label_cell_id: str | None
    value_cell_id: str | None
    label_span_ids: list[str]
    value_span_ids: list[str]
    label_method: Literal["vocabulary-alias-v1"] = LABEL_RESOLUTION_V1
    pairing_method: Literal["caller-supplied-pair-v1", "column-header-below-v1"]
    pairing_id: str | None = None
    status: _Status
    reasons: list[str]
    missing_context: list[str]


class SuppliedPair(BaseModel):
    """An explicit label/value pair from the caller.

    Span ids, when present, are read from the document. The stored span text
    is the original wording. Caller text that disagrees with those spans is
    a conflict.
    """

    model_config = ConfigDict(extra="forbid")

    label_text: str | None = None
    value_text: str | None = None
    label_span_ids: list[str] = Field(default_factory=list)
    value_span_ids: list[str] = Field(default_factory=list)
    label_line_ids: list[str] = Field(default_factory=list)
    value_line_ids: list[str] = Field(default_factory=list)
    page_number: int | None = None
    table_id: str | None = None
    label_cell_id: str | None = None
    value_cell_id: str | None = None


class ContextClaim(BaseModel):
    """A caller-supplied link from source spans to one interpretation.

    This is not automatic page context. Span ids are checked against the
    document. ``interpretation`` is the caller's reading of ``original`` text.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    span_ids: list[str] = Field(min_length=1)


def normalize_observations(
    document: Document,
    tables: Sequence[TableCandidate] = (),
    relationships: Sequence[StructuralRelationship] = (),
    *,
    pairs: Sequence[SuppliedPair] = (),
    context: Mapping[str, str] | None = None,
    claims: Sequence[ContextClaim] = (),
    vocabulary: Vocabulary | None = None,
) -> list[NormalizedObservation]:
    """Return derived observations. Inputs are not modified."""
    selected = initial_vocabulary() if vocabulary is None else vocabulary
    supplied = dict(context or {})
    supplied_claims = list(claims)
    observations = [
        _from_pair(document, pair, supplied, supplied_claims, selected)
        for pair in pairs
    ]
    for table in tables:
        observations.extend(
            _from_table(
                document, table, relationships, supplied, supplied_claims, selected
            )
        )
    return observations


def _from_pair(
    document: Document,
    pair: SuppliedPair,
    context: dict[str, str],
    claims: Sequence[ContextClaim],
    vocabulary: Vocabulary,
) -> NormalizedObservation:
    label, label_ids, label_reasons = _pair_text(
        document, pair.label_span_ids, pair.label_line_ids, pair.label_text
    )
    value, value_ids, value_reasons = _pair_text(
        document, pair.value_span_ids, pair.value_line_ids, pair.value_text
    )
    return _observe(
        document=document,
        label=label,
        value=value,
        context=context,
        vocabulary=vocabulary,
        page_number=pair.page_number,
        table_id=pair.table_id,
        label_cell_id=pair.label_cell_id,
        value_cell_id=pair.value_cell_id,
        label_span_ids=label_ids,
        value_span_ids=value_ids,
        pairing_method=PAIR_CALLER_V1,
        pairing_id=None,
        relationship=None,
        extra_reasons=label_reasons + value_reasons,
        blocked=[],
        claims=claims,
    )


def _from_table(
    document: Document,
    table: TableCandidate,
    relationships: Sequence[StructuralRelationship],
    context: dict[str, str],
    claims: Sequence[ContextClaim],
    vocabulary: Vocabulary,
) -> list[NormalizedObservation]:
    by_slot: dict[tuple[int, int], list[TableCell]] = {}
    for cell in table.cells:
        if cell.row is None or cell.column is None:
            continue
        by_slot.setdefault((cell.row, cell.column), []).append(cell)
    headers: dict[int, list[tuple[int, StructuralRelationship]]] = {}
    for relationship in relationships:
        if relationship.type != "column-header":
            continue
        slot = _cell_slot(table.id, relationship.source.id)
        if slot is None:
            continue
        headers.setdefault(slot[1], []).append((slot[0], relationship))
    observations: list[NormalizedObservation] = []
    for column, column_headers in headers.items():
        ordered = sorted(column_headers, key=lambda item: item[0])
        for index, (row, relationship) in enumerate(ordered):
            following = ordered[index + 1][0] if index + 1 < len(ordered) else None
            observations.extend(
                _below_header(
                    document,
                    table,
                    by_slot,
                    column,
                    row,
                    following,
                    relationship,
                    context,
                    claims,
                    vocabulary,
                )
            )
    return observations


def _below_header(
    document: Document,
    table: TableCandidate,
    by_slot: dict[tuple[int, int], list[TableCell]],
    column: int,
    header_row: int,
    next_header: int | None,
    relationship: StructuralRelationship,
    context: dict[str, str],
    claims: Sequence[ContextClaim],
    vocabulary: Vocabulary,
) -> list[NormalizedObservation]:
    header_cells = by_slot.get((header_row, column), [])
    label, label_ids, label_reasons = _cell_wording(header_cells)
    if len(header_cells) != 1:
        label_reasons = [*label_reasons, "conflicting-cell-slot"]
    blocked = _membership_blocks(table, label_ids)
    observations: list[NormalizedObservation] = []
    value_rows = [
        row
        for row, cell_column in by_slot
        if cell_column == column and row > header_row
        if next_header is None or row < next_header
    ]
    for row in sorted(value_rows):
        value_cells = by_slot[(row, column)]
        value, value_ids, value_reasons = _cell_wording(value_cells)
        reasons = label_reasons + value_reasons
        if len(value_cells) != 1:
            reasons.append("conflicting-cell-slot")
        observations.append(
            _observe(
                document=document,
                label=label,
                value=value,
                context=context,
                vocabulary=vocabulary,
                page_number=table.page_number,
                table_id=table.id,
                label_cell_id=_cell_id(table.id, header_row, column),
                value_cell_id=_cell_id(table.id, row, column),
                label_span_ids=label_ids,
                value_span_ids=value_ids,
                pairing_method=PAIR_HEADER_V1,
                pairing_id=relationship.id,
                relationship=relationship,
                extra_reasons=reasons,
                blocked=blocked + _membership_blocks(table, value_ids),
                claims=claims,
            )
        )
    return observations


def _observe(
    *,
    document: Document,
    label: str,
    value: str,
    context: dict[str, str],
    vocabulary: Vocabulary,
    page_number: int | None,
    table_id: str | None,
    label_cell_id: str | None,
    value_cell_id: str | None,
    label_span_ids: list[str],
    value_span_ids: list[str],
    pairing_method: Literal["caller-supplied-pair-v1", "column-header-below-v1"],
    pairing_id: str | None,
    relationship: StructuralRelationship | None,
    extra_reasons: list[str],
    blocked: list[str],
    claims: Sequence[ContextClaim] = (),
) -> NormalizedObservation:
    resolution = resolve_label(
        label,
        context=context,
        relationship=relationship,
        vocabulary=vocabulary,
    )
    concept_id = None
    if resolution.status == "resolved" and len(resolution.concept_candidates) == 1:
        concept_id = resolution.concept_candidates[0]
    body, prefixes = _split_prefix(value)
    if "multiple-values" in extra_reasons:
        parsed = _Parsed(reasons=["multiple-values"])
    else:
        parsed = _parse_amount(body)
    markers = _label_markers(label)
    bound, claim_reasons = _bind_claims(document, claims)
    qualifiers, qualifier_reasons = _qualifiers(
        vocabulary,
        concept_id,
        label,
        value,
        context,
        resolution.stated_context,
        resolution.matched_alias,
        prefixes,
        [item for item in bound if item[0] != "entity_scope"],
    )
    qualifiers, basis_reasons = _infer_point_in_time(
        vocabulary, concept_id, qualifiers, value, bound
    )
    (
        unit,
        denomination,
        scale,
        marker_reasons,
        unit_evidence,
        denomination_evidence,
        scale_evidence,
    ) = _measure(
        parsed,
        markers,
        context,
        label,
        value,
        resolution.matched_alias,
        vocabulary,
        concept_id,
    )
    scope, scope_evidence, scope_reasons = _scope(
        vocabulary,
        concept_id,
        qualifiers,
        context,
        [item[2] for item in bound if item[0] == "entity_scope"],
    )
    reasons = _unique(
        extra_reasons
        + blocked
        + parsed.reasons
        + marker_reasons
        + qualifier_reasons
        + claim_reasons
        + basis_reasons
        + scope_reasons
        + _label_reasons(resolution.status)
    )
    missing = list(resolution.missing_context)
    if concept_id is not None:
        missing = _missing_required(vocabulary, concept_id, qualifiers, missing)
        if unit is not None and unit not in _concept(vocabulary, concept_id).units:
            reasons = _unique([*reasons, "unsupported-unit"])
        if scope is None:
            missing = _unique([*missing, "entity_scope"])
    if (
        missing
        and concept_id is not None
        and _required_missing(vocabulary, concept_id, qualifiers)
    ):
        reasons = _unique([*reasons, "missing-context"])
    if pairing_method == PAIR_HEADER_V1:
        reasons = _unique([*reasons, "candidate-header"])
    if _needs_unit(unit, denomination, reasons, parsed.reasons):
        reasons = _unique([*reasons, "missing-unit"])
    status = _status(pairing_method, reasons)
    return NormalizedObservation(
        concept_id=concept_id,
        concept_candidates=list(resolution.concept_candidates),
        vocabulary_version=resolution.vocabulary_version,
        original_label=label,
        original_value=value,
        amount=parsed.amount,
        unit=unit,
        denomination=denomination,
        scale=scale,
        unit_evidence=unit_evidence,
        denomination_evidence=denomination_evidence,
        scale_evidence=scale_evidence,
        entity_scope=scope,
        scope_evidence=scope_evidence,
        qualifiers=qualifiers,
        source_sha256=document.source_sha256,
        page_number=page_number,
        table_id=table_id,
        label_cell_id=label_cell_id,
        value_cell_id=value_cell_id,
        label_span_ids=label_span_ids,
        value_span_ids=value_span_ids,
        label_method=LABEL_RESOLUTION_V1,
        pairing_method=pairing_method,
        pairing_id=pairing_id,
        status=status,
        reasons=reasons,
        missing_context=missing,
    )


def _status(
    pairing_method: Literal["caller-supplied-pair-v1", "column-header-below-v1"],
    reasons: list[str],
) -> _Status:
    blocking = {
        "ambiguous-label",
        "unresolved-label",
        "missing-value",
        "missing-unit",
        "missing-context",
        "missing-span",
        "source-text-mismatch",
        "unsupported-notation",
        "ambiguous-separator",
        "multiple-values",
        "range",
        "inequality",
        "conflicting-unit",
        "conflicting-scale",
        "conflicting-scope",
        "conflicting-qualifier",
        "conflicting-membership",
        "assigned-unassigned",
        "conflicting-cell-slot",
        "unsupported-unit",
        "unsupported-scale",
        "conflicting-basis",
    }
    if pairing_method == PAIR_HEADER_V1:
        allowed = {
            "candidate-header",
            "missing-context",
            "missing-unit",
        }
        if reasons and all(reason in allowed for reason in reasons):
            return "candidate"
        return "unresolved"
    if any(reason in blocking for reason in reasons):
        return "unresolved"
    return "resolved"


class _Parsed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: Decimal | None = None
    unit: str | None = None
    scale: str | None = None
    unit_evidence: list[SourcedValue] = Field(default_factory=list)
    scale_evidence: list[SourcedValue] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def _parse_amount(text: str) -> _Parsed:
    stripped = unicodedata.normalize("NFC", text).strip()
    if _is_missing(stripped):
        return _Parsed(reasons=["missing-value"])
    if _INEQUALITY.search(stripped) is not None:
        return _Parsed(reasons=["inequality"])
    if _RANGE.search(stripped) is not None or _AND.search(stripped) is not None:
        reason = "range" if _RANGE.search(stripped) else "multiple-values"
        return _Parsed(reasons=[reason])
    if stripped.count("%") > 1:
        return _Parsed(reasons=["multiple-values"])
    working = stripped
    unit_evidence: list[SourcedValue] = []
    scale_evidence: list[SourcedValue] = []
    reasons: list[str] = []
    percent = _PERCENT_SUFFIX.search(working)
    if percent is not None:
        unit_evidence.append(
            SourcedValue(value="percent", origin="value", source_text=percent.group(0))
        )
        working = working[: percent.start()]
    if "%" in working or _PERCENT_WORD.search(working) is not None:
        return _Parsed(reasons=["unsupported-notation"])
    working, found_units, found_scales = _remove_markers(working, "value")
    if len({item.value for item in found_units}) > 1:
        reasons.append("conflicting-unit")
    if len({item.value for item in found_scales}) > 1:
        reasons.append("conflicting-scale")
    unit_evidence.extend(found_units)
    scale_evidence.extend(found_scales)
    token = "".join(working.split())
    amount = None if reasons else _decimal_token(token)
    if amount is None and "missing-value" not in reasons:
        if "," in token:
            reasons.append("ambiguous-separator")
        elif not reasons:
            reasons.append("unsupported-notation")
    unit_values = {item.value for item in unit_evidence}
    scale_values = {item.value for item in scale_evidence}
    unit = unit_evidence[0].value if len(unit_values) == 1 else None
    scale = scale_evidence[0].value if len(scale_values) == 1 else None
    if reasons:
        unit = None if "conflicting-unit" in reasons else unit
        scale = None if "conflicting-scale" in reasons else scale
        amount = None
    return _Parsed(
        amount=amount,
        unit=unit,
        scale=scale,
        unit_evidence=unit_evidence,
        scale_evidence=scale_evidence,
        reasons=_unique(reasons),
    )


def _label_markers(label: str) -> _Parsed:
    text = unicodedata.normalize("NFC", label)
    unit_evidence: list[SourcedValue] = []
    if "%" in text or _PERCENT_WORD.search(text) is not None:
        unit_evidence.append(
            SourcedValue(value="percent", origin="label", source_text=label)
        )
    _unused, currencies, scales = _remove_markers(text, "label")
    return _Parsed(
        unit_evidence=[*unit_evidence, *currencies],
        scale_evidence=scales,
    )


def _remove_markers(
    text: str, origin: _Origin
) -> tuple[str, list[SourcedValue], list[SourcedValue]]:
    currencies: list[SourcedValue] = []
    scales: list[SourcedValue] = []
    for pattern, value in _CURRENCY:
        text, matches = _excise(text, pattern)
        currencies.extend(
            SourcedValue(value=value, origin=origin, source_text=match)
            for match in matches
        )
    for pattern, value in _SCALE:
        text, matches = _excise(text, pattern)
        scales.extend(
            SourcedValue(value=value, origin=origin, source_text=match)
            for match in matches
        )
    return text, currencies, scales


def _excise(text: str, pattern: re.Pattern[str]) -> tuple[str, list[str]]:
    matches = [match.group(0) for match in pattern.finditer(text)]
    return pattern.sub(" ", text), matches


def _decimal_token(token: str) -> Decimal | None:
    if token == "" or token in {"+", "-"}:
        return None
    sign = ""
    body = token
    if body[0] in "+-":
        sign = "-" if body[0] == "-" else ""
        body = body[1:]
    if body.count(".") > 1 or not body or body[0] in "+-":
        return None
    integer, dot, fraction = body.partition(".")
    if dot and (fraction == "" or not fraction.isdigit()):
        return None
    if not _grouping_ok(integer):
        return None
    digits = integer.replace(",", "")
    number = digits if not dot else f"{digits}.{fraction}"
    return Decimal(f"{sign}{number}")


def _grouping_ok(integer: str) -> bool:
    if integer == "":
        return False
    if "," not in integer:
        return integer.isdigit()
    parts = integer.split(",")
    if any(part == "" or not part.isdigit() for part in parts):
        return False
    western = len(parts[0]) in {1, 2, 3} and all(len(part) == 3 for part in parts[1:])
    indian = (
        len(parts[-1]) == 3
        and len(parts[0]) in {1, 2}
        and all(len(part) == 2 for part in parts[1:-1])
    )
    return western or indian


def _is_missing(text: str) -> bool:
    folded = text.casefold().strip()
    if folded == "":
        return True
    if folded != "" and folded.strip("-–—") == "":
        return True
    return folded in {"n/a", "na", "n.a", "n.a."}


def _measure(
    parsed: _Parsed,
    label_markers: _Parsed,
    context: dict[str, str],
    label: str,
    value: str,
    matched_alias: str | None,
    vocabulary: Vocabulary,
    concept_id: str | None,
) -> tuple[
    str | None,
    str | None,
    str | None,
    list[str],
    list[SourcedValue],
    list[SourcedValue],
    list[SourcedValue],
]:
    reasons: list[str] = []
    kind_evidence: list[SourcedValue] = []
    denomination_evidence: list[SourcedValue] = []
    scale_evidence = [*parsed.scale_evidence, *label_markers.scale_evidence]
    for item in [*parsed.unit_evidence, *label_markers.unit_evidence]:
        if item.value == "INR":
            denomination_evidence.append(item)
            kind_evidence.append(
                SourcedValue(
                    value="currency", origin=item.origin, source_text=item.source_text
                )
            )
        else:
            kind_evidence.append(item)
    context_unit = context.get("unit")
    if context_unit is not None:
        kind, denomination = _canonical_unit(context_unit)
        if kind is None:
            reasons.append("unsupported-unit")
        else:
            kind_evidence.append(
                SourcedValue(value=kind, origin="context", source_text=context_unit)
            )
            if denomination is not None:
                denomination_evidence.append(
                    SourcedValue(
                        value=denomination,
                        origin="context",
                        source_text=context_unit,
                    )
                )
    context_scale = context.get("scale")
    if context_scale is not None:
        canonical = _canonical_scale(context_scale)
        if canonical is None:
            reasons.append("unsupported-scale")
        else:
            scale_evidence.append(
                SourcedValue(
                    value=canonical, origin="context", source_text=context_scale
                )
            )
    per_unit = _per_unit_evidence(label, value, matched_alias, context)
    defined = _defined_unit(vocabulary, concept_id)
    if defined is not None:
        per_unit.append(defined)
    if per_unit and "percent" not in {item.value for item in kind_evidence}:
        kind_evidence = [item for item in kind_evidence if item.value != "currency"]
        kind_evidence.extend(per_unit)
    elif per_unit:
        kind_evidence.extend(per_unit)
    kinds = {item.value for item in kind_evidence}
    unit = next(iter(kinds)) if len(kinds) == 1 else None
    if unit is None and kinds:
        reasons.append("conflicting-unit")
    denominations = {item.value for item in denomination_evidence}
    denomination = next(iter(denominations)) if len(denominations) == 1 else None
    if denomination is None and denomination_evidence:
        reasons.append("conflicting-unit")
    scale = _one_value(scale_evidence)
    if scale is None and scale_evidence:
        reasons.append("conflicting-scale")
    return (
        unit,
        denomination,
        scale,
        reasons,
        kind_evidence,
        denomination_evidence,
        scale_evidence,
    )


def _canonical_unit(text: str) -> tuple[str | None, str | None]:
    folded = text.casefold().strip()
    if folded == "percent":
        return "percent", None
    if folded == "fraction":
        return "fraction", None
    if folded == "currency":
        return "currency", None
    if folded in {"currency_per_unit", "currency per unit"}:
        return "currency_per_unit", None
    if folded == "inr":
        return "currency", "INR"
    return None, None


def _per_unit_evidence(
    label: str,
    value: str,
    matched_alias: str | None,
    context: dict[str, str],
) -> list[SourcedValue]:
    found: list[SourcedValue] = []
    for text, origin in ((label, "label"), (value, "value")):
        if "per unit" in normalize_label(text):
            found.append(
                SourcedValue(value="currency_per_unit", origin=origin, source_text=text)
            )
    if matched_alias is not None and "per unit" in normalize_label(matched_alias):
        found.append(
            SourcedValue(
                value="currency_per_unit",
                origin="alias",
                source_text=matched_alias,
            )
        )
    raw = context.get("unit")
    if raw is not None and raw.casefold().strip() in {
        "currency_per_unit",
        "currency per unit",
    }:
        found.append(
            SourcedValue(value="currency_per_unit", origin="context", source_text=raw)
        )
    return found


def _defined_unit(
    vocabulary: Vocabulary, concept_id: str | None
) -> SourcedValue | None:
    if concept_id is None:
        return None
    for rule in vocabulary.rules:
        if rule.concept_id == concept_id and rule.effect == "currency_per_unit":
            return SourcedValue(
                value="currency_per_unit",
                origin="concept-definition",
                source_text=rule.id,
                method="concept-definition-v1",
                interpretation="currency_per_unit",
            )
    return None


def _needs_unit(
    unit: str | None,
    denomination: str | None,
    reasons: list[str],
    parsed_reasons: list[str],
) -> bool:
    if "missing-value" in reasons or "conflicting-unit" in reasons:
        return False
    if parsed_reasons or "multiple-values" in reasons:
        return False
    if unit is None:
        return True
    return unit in {"currency", "currency_per_unit"} and denomination is None


def _scope(
    vocabulary: Vocabulary,
    concept_id: str | None,
    qualifiers: list[ObservationQualifier],
    context: dict[str, str],
    scope_claims: list[SourcedValue],
) -> tuple[str | None, list[SourcedValue], list[str]]:
    if concept_id is None:
        return None, [], []
    allowed = set(_concept(vocabulary, concept_id).entity_scope)
    tokens: list[str] = []
    evidence: list[SourcedValue] = []
    raw = context.get("entity_scope")
    if raw is not None:
        tokens.append(raw)
        evidence.append(
            SourcedValue(
                value=raw,
                origin="context",
                source_text=raw,
                interpretation=raw,
                method="caller-context-v1",
            )
        )
    for item in scope_claims:
        tokens.append(item.interpretation or item.value)
        evidence.append(item)
    for qualifier in qualifiers:
        if qualifier.id == "subject" and qualifier.value in allowed:
            tokens.append(qualifier.value)
            evidence.extend(qualifier.evidence)
    unique = list(dict.fromkeys(tokens))
    if any(token not in allowed for token in unique) or len(unique) > 1:
        return None, evidence, ["conflicting-scope"]
    if len(unique) == 1:
        return unique[0], evidence, []
    return None, evidence, []


def _canonical_scale(text: str) -> str | None:
    folded = text.casefold().strip()
    if folded == "lakh":
        return "lakh"
    if folded == "crore":
        return "crore"
    return None


def _one_value(evidence: list[SourcedValue]) -> str | None:
    values = {item.value for item in evidence}
    if len(values) == 1:
        return next(iter(values))
    return None


def _qualifiers(
    vocabulary: Vocabulary,
    concept_id: str | None,
    label: str,
    value: str,
    context: dict[str, str],
    stated: dict[str, str],
    matched_alias: str | None,
    prefixes: list[tuple[str, str, str]],
    claims: list[tuple[str, str, SourcedValue]],
) -> tuple[list[ObservationQualifier], list[str]]:
    if concept_id is None:
        return [], []
    allowed = set(_qualifier_ids(vocabulary, concept_id))
    found: list[tuple[str, str, SourcedValue]] = []
    for text, origin in ((label, "label"), (value, "value")):
        normalized = normalize_label(text)
        for phrase, qualifier_id, qualifier_value in _PHRASES:
            if qualifier_id not in allowed:
                continue
            if phrase in normalized:
                found.append(
                    (
                        qualifier_id,
                        qualifier_value,
                        SourcedValue(
                            value=qualifier_value, origin=origin, source_text=text
                        ),
                    )
                )
    for qualifier_id, qualifier_value, source in prefixes:
        if qualifier_id not in allowed:
            continue
        found.append(
            (
                qualifier_id,
                qualifier_value,
                SourcedValue(value=qualifier_value, origin="value", source_text=source),
            )
        )
    alias_text = matched_alias if matched_alias is not None else label
    for qualifier_id, qualifier_value in stated.items():
        if qualifier_id not in allowed:
            continue
        found.append(
            (
                qualifier_id,
                qualifier_value,
                SourcedValue(
                    value=qualifier_value, origin="alias", source_text=alias_text
                ),
            )
        )
    for qualifier_id, qualifier_value in context.items():
        if qualifier_id in _MARKER_KEYS or qualifier_id not in allowed:
            continue
        found.append(
            (
                qualifier_id,
                qualifier_value,
                SourcedValue(
                    value=qualifier_value,
                    origin="context",
                    source_text=qualifier_value,
                    interpretation=qualifier_value,
                    method="caller-context-v1",
                ),
            )
        )
    for qualifier_id, qualifier_value, evidence in claims:
        if qualifier_id not in allowed:
            continue
        found.append((qualifier_id, qualifier_value, evidence))
    return _group_qualifiers(found)


def _bind_claims(
    document: Document, claims: Sequence[ContextClaim]
) -> tuple[list[tuple[str, str, SourcedValue]], list[str]]:
    bound: list[tuple[str, str, SourcedValue]] = []
    reasons: list[str] = []
    for claim in claims:
        texts: list[str] = []
        ids: list[str] = []
        valid = True
        for span_id in claim.span_ids:
            located = _locate_span(document, span_id)
            if located is None or located[0] != claim.page_number:
                reasons.append("missing-span")
                valid = False
                break
            texts.append(located[1].text.strip())
            ids.append(span_id)
        if not valid:
            continue
        bound.append(
            (
                claim.id,
                claim.interpretation,
                SourcedValue(
                    value=claim.interpretation,
                    origin="context",
                    source_text=" ".join(texts),
                    document_sha256=document.source_sha256,
                    page_number=claim.page_number,
                    span_ids=ids,
                    interpretation=claim.interpretation,
                    method="caller-source-v1",
                ),
            )
        )
    return bound, _unique(reasons)


def _infer_point_in_time(
    vocabulary: Vocabulary,
    concept_id: str | None,
    qualifiers: list[ObservationQualifier],
    value: str,
    bound: list[tuple[str, str, SourcedValue]],
) -> tuple[list[ObservationQualifier], list[str]]:
    rule = next(
        (
            item
            for item in vocabulary.rules
            if concept_id is not None
            and item.concept_id == concept_id
            and item.effect == "point_in_time"
        ),
        None,
    )
    if rule is None:
        return qualifiers, []
    triggering = [
        evidence
        for claim_id, _interpretation, evidence in bound
        if claim_id == "as_of_date"
    ]
    if not triggering:
        return qualifiers, []
    claim_text = normalize_label(" ".join(item.source_text for item in triggering))
    value_text = normalize_label(value)
    as_on = "as on" in claim_text or "as at" in claim_text
    average = "average" in claim_text or "average" in value_text or "aaum" in claim_text
    if as_on and average:
        return qualifiers, ["conflicting-basis"]
    if not as_on or average:
        return qualifiers, []
    evidence = SourcedValue(
        value="point_in_time",
        origin="concept-definition",
        source_text=rule.id,
        document_sha256=triggering[0].document_sha256,
        page_number=triggering[0].page_number,
        span_ids=list(triggering[0].span_ids),
        interpretation="point_in_time",
        method="concept-definition-v1",
    )
    current = [item for item in qualifiers if item.id == "aum_basis"]
    if current and current[0].value != "point_in_time":
        return qualifiers, ["conflicting-basis"]
    if current:
        return [
            item.model_copy(update={"evidence": [*item.evidence, evidence]})
            if item.id == "aum_basis"
            else item
            for item in qualifiers
        ], []
    added = ObservationQualifier(
        id="aum_basis", value="point_in_time", evidence=[evidence]
    )
    return [*qualifiers, added], []


def _locate_span(document: Document, span_id: str) -> tuple[int, TextSpan] | None:
    for page in document.pages:
        for span in page.spans:
            if span.id == span_id:
                return page.number, span
    return None


def _group_qualifiers(
    found: list[tuple[str, str, SourcedValue]],
) -> tuple[list[ObservationQualifier], list[str]]:
    grouped: dict[str, list[tuple[str, SourcedValue]]] = {}
    for qualifier_id, qualifier_value, evidence in found:
        grouped.setdefault(qualifier_id, []).append((qualifier_value, evidence))
    qualifiers: list[ObservationQualifier] = []
    conflict = False
    for qualifier_id in sorted(grouped):
        values = {value for value, _evidence in grouped[qualifier_id]}
        evidence = [item for _value, item in grouped[qualifier_id]]
        if len(values) != 1:
            conflict = True
            continue
        qualifiers.append(
            ObservationQualifier(
                id=qualifier_id,
                value=next(iter(values)),
                evidence=evidence,
            )
        )
    reasons = ["conflicting-qualifier"] if conflict else []
    return qualifiers, reasons


def _missing_required(
    vocabulary: Vocabulary,
    concept_id: str,
    qualifiers: list[ObservationQualifier],
    label_missing: list[str],
) -> list[str]:
    present = {item.id for item in qualifiers}
    concept = _concept(vocabulary, concept_id)
    ordered = [item.id for item in concept.qualifiers if item.id not in present]
    carried = [item for item in label_missing if item not in present]
    return _unique([*ordered, *carried])


def _required_missing(
    vocabulary: Vocabulary,
    concept_id: str,
    qualifiers: list[ObservationQualifier],
) -> bool:
    present = {item.id for item in qualifiers}
    concept = _concept(vocabulary, concept_id)
    return any(
        item.role == "required" and item.id not in present
        for item in concept.qualifiers
    )


def _qualifier_ids(vocabulary: Vocabulary, concept_id: str) -> list[str]:
    return [item.id for item in _concept(vocabulary, concept_id).qualifiers]


def _concept(vocabulary: Vocabulary, concept_id: str):
    for concept in vocabulary.concepts:
        if concept.id == concept_id:
            return concept
    raise ValueError(f"unknown concept {concept_id}")


def _label_reasons(status: str) -> list[str]:
    if status == "ambiguous":
        return ["ambiguous-label"]
    if status == "unresolved":
        return ["unresolved-label"]
    return []


def _pair_text(
    document: Document,
    span_ids: list[str],
    line_ids: list[str],
    supplied: str | None,
) -> tuple[str, list[str], list[str]]:
    if not span_ids and not line_ids:
        return ("" if supplied is None else supplied), [], []
    selected, reasons = _selected_ids(document, span_ids, line_ids)
    if "missing-span" in reasons:
        return supplied or "", selected, reasons
    joined, ordered, compose_reasons = _compose(document, selected)
    reasons = _unique([*reasons, *compose_reasons])
    if supplied is not None and supplied != joined:
        reasons.append("source-text-mismatch")
    return joined, ordered, reasons


def _selected_ids(
    document: Document,
    span_ids: list[str],
    line_ids: list[str],
) -> tuple[list[str], list[str]]:
    lines = {line.id: line for page in document.pages for line in page.lines}
    if line_ids and not span_ids:
        selected: list[str] = []
        reasons: list[str] = []
        for line_id in line_ids:
            line = lines.get(line_id)
            if line is None:
                reasons.append("missing-span")
                continue
            selected.extend(line.span_ids)
        return selected, _unique(reasons)
    if not line_ids:
        return list(span_ids), []
    line_of = _line_index(document)
    grouped = {line_id: [] for line_id in line_ids}
    reasons = []
    for span_id in span_ids:
        line = line_of.get(span_id)
        if line is None or line.id not in grouped:
            reasons.append("missing-span")
            continue
        grouped[line.id].append(span_id)
    selected = [span_id for line_id in line_ids for span_id in grouped[line_id]]
    return selected, _unique(reasons)


def _compose(
    document: Document, span_ids: list[str]
) -> tuple[str, list[str], list[str]]:
    spans = _span_index(document)
    line_of = _line_index(document)
    grouped: dict[str, list[tuple[int, TextSpan]]] = {}
    order: list[str] = []
    reasons: list[str] = []
    for index, span_id in enumerate(span_ids):
        span = spans.get(span_id)
        if span is None:
            return "", list(span_ids), ["missing-span"]
        line = line_of.get(span_id)
        key = line.id if line is not None else span_id
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((index, span))
    ordered: list[str] = []
    parts: list[str] = []
    numeric = 0
    for key in order:
        line = next(
            (item for item in line_of.values() if item.id == key),
            None,
        )
        members = grouped[key]
        if line is not None and _horizontal(line):
            members = sorted(
                members,
                key=lambda item: (item[1].origin.x, item[1].bbox.x0, item[1].id),
            )
        else:
            members = sorted(members, key=lambda item: item[0])
        for _index, span in members:
            ordered.append(span.id)
            if _numeric_fragment(span.text):
                numeric += 1
            stripped = span.text.strip()
            if stripped:
                parts.append(stripped)
    if numeric > 1:
        reasons.append("multiple-values")
    return " ".join(parts), ordered, reasons


def _numeric_fragment(text: str) -> bool:
    return any(character.isdigit() for character in text)


def _horizontal(line: TextLine) -> bool:
    if line.writing_mode != 0:
        return False
    dx, dy = line.direction
    return (
        abs(dx - 1) <= _HORIZONTAL_DIRECTION_TOLERANCE
        and abs(dy) <= _HORIZONTAL_DIRECTION_TOLERANCE
    )


def _span_index(document: Document) -> dict[str, TextSpan]:
    return {span.id: span for page in document.pages for span in page.spans}


def _line_index(document: Document) -> dict[str, TextLine]:
    return {
        span_id: line
        for page in document.pages
        for line in page.lines
        for span_id in line.span_ids
    }


def _split_prefix(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    match = _QUALIFIER_PREFIX.match(text)
    if match is None:
        return text, []
    mapped = _PREFIX_VALUES.get(match.group(1).casefold())
    if mapped is None:
        return text, []
    qualifier_id, qualifier_value = mapped
    return text[match.end() :].strip(), [
        (qualifier_id, qualifier_value, match.group(0).strip())
    ]


def _cell_wording(cells: list[TableCell]) -> tuple[str, list[str], list[str]]:
    if len(cells) != 1:
        return "", [], ["missing-value"]
    cell = cells[0]
    if cell.lines:
        fragments = [
            (span.span_id, span.text) for line in cell.lines for span in line.spans
        ]
    else:
        fragments = [(item.span_id, item.text) for item in cell.assignments]
    span_ids = [span_id for span_id, _text in fragments]
    if any(text is None for _span_id, text in fragments):
        return "", span_ids, ["missing-value"]
    texts = [text for _span_id, text in fragments if text is not None and text.strip()]
    if not texts:
        return "", span_ids, ["missing-value"]
    numeric = sum(_numeric_fragment(text) for text in texts)
    reasons = ["multiple-values"] if numeric > 1 else []
    return " ".join(text.strip() for text in texts), span_ids, reasons


def _membership_blocks(table: TableCandidate, span_ids: list[str]) -> list[str]:
    unresolved = {item.span_id for item in table.unresolved_memberships}
    unassigned = set(table.unassigned_span_ids)
    reasons: list[str] = []
    for span_id in span_ids:
        if span_id in unresolved:
            reasons.append("conflicting-membership")
        if span_id in unassigned:
            reasons.append("assigned-unassigned")
    return _unique(reasons)


def _cell_slot(table_id: str, cell_id: str) -> tuple[int, int] | None:
    prefix = f"{table_id}-r"
    if not cell_id.startswith(prefix):
        return None
    row_text, separator, column_text = cell_id[len(prefix) :].partition("c")
    if separator != "c" or not row_text.isdigit() or not column_text.isdigit():
        return None
    return int(row_text), int(column_text)


def _cell_id(table_id: str, row: int, column: int) -> str:
    return f"{table_id}-r{row}c{column}"


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
