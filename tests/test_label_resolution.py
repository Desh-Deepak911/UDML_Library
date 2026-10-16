from udml.semantic import VOCABULARY_VERSION, resolve_label


def test_equivalent_labels_share_a_concept_and_keep_source_wording() -> None:
    labels = (
        "Net Asset Value",
        "NAV",
        "N.A.V.",
        "Net-Asset-Value",
        "  net   asset\tvalue ",
    )
    for label in labels:
        result = resolve_label(label)
        assert result.status == "resolved"
        assert result.concept_candidates == ["net_asset_value"]
        assert result.original_label == label
        assert result.qualified_fact is False
        assert result.vocabulary_version == VOCABULARY_VERSION


def test_ambiguous_and_under_specified_labels_stay_unresolved() -> None:
    ambiguous = resolve_label("Net Assets")
    assert ambiguous.status == "ambiguous"
    assert ambiguous.original_label == "Net Assets"
    assert ambiguous.concept_candidates == [
        "assets_under_management",
        "net_asset_value",
    ]
    assert ambiguous.qualified_fact is False

    abbreviation = resolve_label("AUM")
    assert abbreviation.status == "unresolved"
    assert abbreviation.concept_candidates == ["assets_under_management"]
    assert abbreviation.missing_context == ["aum_basis"]
    assert abbreviation.qualified_fact is False

    average = resolve_label("AAUM")
    assert average.status == "unresolved"
    assert average.concept_candidates == []
    assert average.original_label == "AAUM"
