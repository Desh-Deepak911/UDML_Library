"""Resolve a source label against a curated vocabulary.

A resolved result names a label concept. It is not a fully qualified fact.
A candidate relationship stays candidate when its wording matches a concept.
"""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from udml.semantic.vocabulary import (
    Concept,
    ConceptAlias,
    Vocabulary,
    initial_vocabulary,
    normalize_label,
)
from udml.structure.relations import StructuralRelationship

LABEL_RESOLUTION_V1: Literal["vocabulary-alias-v1"] = "vocabulary-alias-v1"

_Status = Literal["resolved", "ambiguous", "unresolved"]


class _IndexedAlias(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    alias: ConceptAlias


class LabelResolution(BaseModel):
    """The outcome of matching one source label.

    ``original_label`` is the caller's text. ``status`` describes the concept
    match only. ``qualified_fact`` stays false: missing qualifiers, ambiguous
    wording, and a candidate header do not become an established fact.
    """

    model_config = ConfigDict(extra="forbid")

    original_label: str
    status: _Status
    matched_alias: str | None
    concept_candidates: list[str]
    method: Literal["vocabulary-alias-v1"] = LABEL_RESOLUTION_V1
    vocabulary_version: str
    missing_context: list[str]
    stated_context: dict[str, str] = Field(default_factory=dict)
    relationship_id: str | None = None
    relationship_status: Literal["structural", "candidate"] | None = None
    qualified_fact: Literal[False] = False


def resolve_label(
    label: str,
    *,
    context: Mapping[str, str] | None = None,
    relationship: StructuralRelationship | None = None,
    vocabulary: Vocabulary | None = None,
) -> LabelResolution:
    """Match ``label`` using explicit context and an optional relationship.

    ``context`` is used only as supplied. Unknown keys do not select a
    concept and are not stored as qualifiers. ``vocabulary`` defaults to the
    built-in list. Neither argument is modified.
    """
    selected = initial_vocabulary() if vocabulary is None else vocabulary
    supplied = dict(context or {})
    full, partial = _classify(_index(selected), normalize_label(label), supplied)
    relationship_id = None if relationship is None else relationship.id
    relationship_status = None if relationship is None else relationship.status
    if len(full) == 1 and not full[0].alias.ambiguous and not partial:
        return _resolved(
            label,
            full[0],
            selected,
            supplied,
            relationship_id,
            relationship_status,
        )
    if full:
        return _from_matches(
            label,
            "ambiguous",
            full,
            partial,
            selected.version,
            relationship_id,
            relationship_status,
        )
    return _from_matches(
        label,
        "unresolved",
        full,
        partial,
        selected.version,
        relationship_id,
        relationship_status,
    )


def _index(vocabulary: Vocabulary) -> dict[str, list[_IndexedAlias]]:
    indexed: dict[str, list[_IndexedAlias]] = {}
    for concept in vocabulary.concepts:
        for alias in concept.aliases:
            key = normalize_label(alias.text)
            indexed.setdefault(key, []).append(
                _IndexedAlias(concept_id=concept.id, alias=alias)
            )
    return indexed


def _classify(
    indexed: dict[str, list[_IndexedAlias]],
    normalized: str,
    context: dict[str, str],
) -> tuple[list[_IndexedAlias], list[tuple[_IndexedAlias, list[str]]]]:
    full: list[_IndexedAlias] = []
    partial: list[tuple[_IndexedAlias, list[str]]] = []
    for entry in indexed.get(normalized, []):
        if _conflicts(entry.alias, context):
            continue
        missing = [key for key in entry.alias.requires if key not in context]
        if missing:
            partial.append((entry, missing))
        else:
            full.append(entry)
    return full, partial


def _conflicts(alias: ConceptAlias, context: dict[str, str]) -> bool:
    for key, value in {**alias.stated_context, **alias.requires}.items():
        if key in context and context[key] != value:
            return True
    return False


def _resolved(
    label: str,
    entry: _IndexedAlias,
    vocabulary: Vocabulary,
    context: dict[str, str],
    relationship_id: str | None,
    relationship_status: Literal["structural", "candidate"] | None,
) -> LabelResolution:
    concept = _concept(vocabulary, entry.concept_id)
    satisfied = set(context) | set(entry.alias.stated_context)
    missing = [
        qualifier.id
        for qualifier in concept.qualifiers
        if qualifier.id not in satisfied
    ]
    return LabelResolution(
        original_label=label,
        status="resolved",
        matched_alias=entry.alias.text,
        concept_candidates=[entry.concept_id],
        vocabulary_version=vocabulary.version,
        missing_context=missing,
        stated_context=dict(entry.alias.stated_context),
        relationship_id=relationship_id,
        relationship_status=relationship_status,
    )


def _from_matches(
    label: str,
    status: Literal["ambiguous", "unresolved"],
    full: list[_IndexedAlias],
    partial: list[tuple[_IndexedAlias, list[str]]],
    version: str,
    relationship_id: str | None,
    relationship_status: Literal["structural", "candidate"] | None,
) -> LabelResolution:
    entries = [entry for entry in full]
    entries.extend(entry for entry, _missing in partial)
    texts = {entry.alias.text for entry in entries}
    missing = sorted({key for _entry, keys in partial for key in keys})
    return LabelResolution(
        original_label=label,
        status=status,
        matched_alias=None if len(texts) != 1 else next(iter(texts)),
        concept_candidates=sorted({entry.concept_id for entry in entries}),
        vocabulary_version=version,
        missing_context=missing,
        relationship_id=relationship_id,
        relationship_status=relationship_status,
    )


def _concept(vocabulary: Vocabulary, concept_id: str) -> Concept:
    for concept in vocabulary.concepts:
        if concept.id == concept_id:
            return concept
    raise ValueError(f"unknown concept {concept_id}")
