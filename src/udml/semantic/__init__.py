"""Label concepts and conservative alias resolution."""

from udml.semantic.normalize import (
    PAIR_CALLER_V1,
    PAIR_HEADER_V1,
    ContextClaim,
    NormalizedObservation,
    ObservationQualifier,
    SourcedValue,
    SuppliedPair,
    normalize_observations,
)
from udml.semantic.resolve import (
    LABEL_RESOLUTION_V1,
    LabelResolution,
    resolve_label,
)
from udml.semantic.vocabulary import (
    VOCABULARY_VERSION,
    Concept,
    ConceptAlias,
    ConceptRule,
    Qualifier,
    Vocabulary,
    initial_vocabulary,
    normalize_label,
)

__all__ = [
    "LABEL_RESOLUTION_V1",
    "PAIR_CALLER_V1",
    "PAIR_HEADER_V1",
    "VOCABULARY_VERSION",
    "Concept",
    "ConceptAlias",
    "ConceptRule",
    "ContextClaim",
    "LabelResolution",
    "NormalizedObservation",
    "ObservationQualifier",
    "Qualifier",
    "SourcedValue",
    "SuppliedPair",
    "Vocabulary",
    "initial_vocabulary",
    "normalize_label",
    "normalize_observations",
    "resolve_label",
]
