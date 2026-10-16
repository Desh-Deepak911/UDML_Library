"""Curated label concepts. This version is not the UDML schema version.

Aliases are explicit. A shared alias is either marked ambiguous or limited
by context that cannot apply to two concepts at once. Construction does not
choose a concept by list order.
"""

import re
import unicodedata
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

VOCABULARY_VERSION = "0.2.0"

_ValueKind = Literal[
    "monetary_amount",
    "ratio",
    "per_unit_amount",
    "portfolio_share",
]
_QualifierRole = Literal["required", "relevant"]


class ConceptAlias(BaseModel):
    """One curated wording for a concept.

    ``requires`` must be supplied by the caller before the alias can match.
    ``stated_context`` is context written into this wording, not a guess
    about other labels. ``ambiguous`` means the wording is shared and must
    not resolve to this concept by itself.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    requires: dict[str, str] = Field(default_factory=dict)
    stated_context: dict[str, str] = Field(default_factory=dict)
    ambiguous: bool = False


class ConceptRule(BaseModel):
    """A named, versioned inference. It is not text found in the document."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    effect: Literal["currency_per_unit", "point_in_time"]
    description: str = Field(min_length=1)


class Qualifier(BaseModel):
    """Context that qualifies a concept beyond the label wording."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    role: _QualifierRole


class Concept(BaseModel):
    """One vocabulary concept and the aliases that may name it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    aliases: list[ConceptAlias] = Field(min_length=1)
    value_kind: _ValueKind
    units: list[str] = Field(min_length=1)
    entity_scope: list[str] = Field(min_length=1)
    qualifiers: list[Qualifier] = Field(default_factory=list)


class Vocabulary(BaseModel):
    """A validated set of concepts at one vocabulary version."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    concepts: list[Concept] = Field(min_length=1)
    rules: list[ConceptRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def aliases_do_not_collide(self) -> Self:
        _validate_aliases(self.concepts)
        return self


def normalize_label(text: str) -> str:
    """Return the match key for a label.

    The key uses NFC, case folding, and collapsed whitespace. A hyphen
    between alphanumeric characters becomes a space. Periods between word
    characters, and one trailing period, are removed. Other punctuation,
    including footnote marks, stays in the key.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    characters: list[str] = []
    for index, character in enumerate(folded):
        if (
            character == "-"
            and index > 0
            and index + 1 < len(folded)
            and folded[index - 1].isalnum()
            and folded[index + 1].isalnum()
        ):
            characters.append(" ")
        else:
            characters.append(character)
    without_hyphen = "".join(characters)
    without_dots = re.sub(r"(?<=\w)\.(?=\w)", "", without_hyphen)
    if without_dots.endswith("."):
        without_dots = without_dots[:-1]
    return " ".join(without_dots.split())


def initial_vocabulary() -> Vocabulary:
    """Return the built-in concept list."""
    return Vocabulary(
        version=VOCABULARY_VERSION,
        concepts=list(_INITIAL_CONCEPTS),
        rules=list(_INITIAL_RULES),
    )


def _validate_aliases(concepts: list[Concept]) -> None:
    concept_ids = [concept.id for concept in concepts]
    if len(concept_ids) != len(set(concept_ids)):
        raise ValueError("concept ids must be unique")
    buckets: dict[str, list[tuple[str, ConceptAlias]]] = {}
    for concept in concepts:
        qualifier_ids = [qualifier.id for qualifier in concept.qualifiers]
        if len(qualifier_ids) != len(set(qualifier_ids)):
            raise ValueError(f"qualifier ids repeat on {concept.id}")
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for alias in concept.aliases:
            normalized = normalize_label(alias.text)
            if not normalized:
                raise ValueError(f"alias on {concept.id} normalizes to empty text")
            _reject_bad_context(concept.id, alias)
            signature = (normalized, tuple(sorted(alias.requires.items())))
            if signature in seen:
                raise ValueError(f"duplicate alias {alias.text!r} on {concept.id}")
            seen.add(signature)
            buckets.setdefault(normalized, []).append((concept.id, alias))
    for normalized, entries in buckets.items():
        _reject_overlapping_aliases(normalized, entries)


def _reject_bad_context(concept_id: str, alias: ConceptAlias) -> None:
    for key, value in {**alias.requires, **alias.stated_context}.items():
        if key == "" or value == "":
            raise ValueError(f"alias {alias.text!r} on {concept_id} has empty context")
    for key in set(alias.requires) & set(alias.stated_context):
        if alias.requires[key] != alias.stated_context[key]:
            raise ValueError(
                f"alias {alias.text!r} on {concept_id} has conflicting context"
            )


def _reject_overlapping_aliases(
    normalized: str,
    entries: list[tuple[str, ConceptAlias]],
) -> None:
    for index, (left_id, left) in enumerate(entries):
        for right_id, right in entries[index + 1 :]:
            if not _compatible(left.requires, right.requires):
                continue
            if left_id == right_id:
                raise ValueError(f"alias {normalized!r} overlaps itself on {left_id}")
            if left.ambiguous and right.ambiguous:
                continue
            raise ValueError(
                f"alias {normalized!r} is shared by {left_id} and {right_id} "
                "without explicit ambiguity or disjoint context"
            )


def _compatible(left: dict[str, str], right: dict[str, str]) -> bool:
    shared = set(left) & set(right)
    return all(left[key] == right[key] for key in shared)


_INITIAL_CONCEPTS: tuple[Concept, ...] = (
    Concept(
        id="assets_under_management",
        definition=(
            "Point-in-time assets managed for a fund or scheme. "
            "Average AUM and AAUM are different measures and are not "
            "this concept."
        ),
        aliases=[
            ConceptAlias(text="assets under management"),
            ConceptAlias(text="asset under management"),
            ConceptAlias(
                text="aum",
                requires={"aum_basis": "point_in_time"},
            ),
            ConceptAlias(text="net assets", ambiguous=True),
        ],
        value_kind="monetary_amount",
        units=["currency"],
        entity_scope=["fund", "scheme"],
        qualifiers=[
            Qualifier(
                id="aum_basis",
                description="Whether the amount is a point-in-time measure.",
                role="required",
            ),
            Qualifier(
                id="as_of_date",
                description="The date of a point-in-time amount.",
                role="relevant",
            ),
        ],
    ),
    Concept(
        id="expense_ratio",
        definition=(
            "The expense ratio of a scheme or plan. Plan type qualifies it. "
            "An as-of date is not an effective date. This is not a return."
        ),
        aliases=[
            ConceptAlias(text="expense ratio"),
            ConceptAlias(text="total expense ratio"),
            ConceptAlias(text="ter"),
        ],
        value_kind="ratio",
        units=["percent", "fraction"],
        entity_scope=["scheme", "plan"],
        qualifiers=[
            Qualifier(
                id="plan",
                description="The plan to which the expense ratio applies.",
                role="required",
            ),
            Qualifier(
                id="effective_date",
                description="The date on which the expense ratio takes effect.",
                role="relevant",
            ),
            Qualifier(
                id="as_of_date",
                description="The disclosure date of the ratio, not its effective date.",
                role="relevant",
            ),
        ],
    ),
    Concept(
        id="net_asset_value",
        definition=(
            "Net asset value per unit. Share class, plan, and option qualify "
            "it when available. The phrase is not total assets under management."
        ),
        aliases=[
            ConceptAlias(text="nav"),
            ConceptAlias(text="net asset value"),
            ConceptAlias(text="net asset value per unit"),
            ConceptAlias(
                text="direct plan nav",
                stated_context={"plan": "direct"},
            ),
            ConceptAlias(text="net assets", ambiguous=True),
        ],
        value_kind="per_unit_amount",
        units=["currency_per_unit"],
        entity_scope=["fund", "scheme", "plan"],
        qualifiers=[
            Qualifier(
                id="share_class",
                description="The share class of the quoted value.",
                role="relevant",
            ),
            Qualifier(
                id="plan",
                description="The plan of the quoted value.",
                role="relevant",
            ),
            Qualifier(
                id="option",
                description="The option, such as growth or income, when stated.",
                role="relevant",
            ),
        ],
    ),
    Concept(
        id="portfolio_weight",
        definition=(
            "The share of a portfolio represented by one holding or category. "
            "This is not a return and not an allocation target."
        ),
        aliases=[
            ConceptAlias(text="portfolio weight"),
            ConceptAlias(text="weight in portfolio"),
            ConceptAlias(text="% of net assets"),
            ConceptAlias(text="percent of net assets"),
            ConceptAlias(text="% of aum"),
            ConceptAlias(text="% to net assets", requires={"subject": "holding"}),
            ConceptAlias(text="% to net assets", requires={"subject": "category"}),
            ConceptAlias(text="weight", requires={"subject": "holding"}),
            ConceptAlias(text="weight", requires={"subject": "category"}),
        ],
        value_kind="portfolio_share",
        units=["percent", "fraction"],
        entity_scope=["portfolio", "holding", "category"],
        qualifiers=[
            Qualifier(
                id="subject",
                description="Whether the weight is for a holding or a category.",
                role="required",
            ),
            Qualifier(
                id="as_of_date",
                description="The portfolio date of the weight.",
                role="relevant",
            ),
        ],
    ),
)


_INITIAL_RULES: tuple[ConceptRule, ...] = (
    ConceptRule(
        id="nav-currency-per-unit-v1",
        concept_id="net_asset_value",
        effect="currency_per_unit",
        description=(
            "A resolved NAV label sets unit kind currency_per_unit. "
            "The source text is not treated as containing the words per unit. "
            "A currency mark remains a denomination."
        ),
    ),
    ConceptRule(
        id="aum-as-on-point-in-time-v1",
        concept_id="assets_under_management",
        effect="point_in_time",
        description=(
            "Caller-supplied As on or As at wording for an AUM value sets "
            "aum_basis point_in_time. Average-period wording blocks the rule."
        ),
    ),
)
