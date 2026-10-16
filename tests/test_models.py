from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from udml.core import BoundingBox, Document, Page, TextSpan
from udml.core.models import build_source_groups

SHA256 = "a" * 64
EXTRACTED_AT = datetime(
    2026, 10, 16, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
)


def _span(**overrides) -> TextSpan:
    values = {
        "id": "p1-b0-l0-s0",
        "text": "Hello",
        "bbox": {"x0": -4.0, "y0": -2.0, "x1": 700.0, "y1": 22.0},
        "font_name": "Helvetica",
        "font_size": 12,
        "font_flags": 0,
        "origin": {"x": 0.0, "y": 10.0},
        "block_index": 0,
        "line_index": 0,
        "span_index": 0,
    }
    values.update(overrides)
    return TextSpan.model_validate(values)


def _page(**overrides) -> Page:
    values = {
        "number": 1,
        "width": 612,
        "height": 792,
        "rotation": 0,
        "spans": [_span()],
    }
    values.update(overrides)
    if "lines" not in values or "blocks" not in values:
        orientations = {
            (span.block_index, span.line_index): ((1.0, 0.0), 0)
            for span in values["spans"]
        }
        lines, blocks = build_source_groups(
            values["number"], values["spans"], orientations
        )
        values.setdefault("lines", lines)
        values.setdefault("blocks", blocks)
    return Page.model_validate(values)


def _document(**overrides) -> Document:
    values = {
        "source_filename": "sample.pdf",
        "source_sha256": SHA256,
        "parser_name": "pymupdf",
        "parser_version": "1.26.4",
        "extracted_at": EXTRACTED_AT,
        "pages": [_page()],
    }
    values.update(overrides)
    return Document.model_validate(values)


def test_valid_minimal_document():
    document = _document(pages=[_page(rotation=90)])
    page = document.pages[0]
    span = page.spans[0]
    box = span.bbox

    assert document.schema_version == "0.5.0"
    assert document.source_filename == "sample.pdf"
    assert document.source_sha256 == SHA256
    assert document.parser_name == "pymupdf"
    assert document.parser_version == "1.26.4"
    assert document.extracted_at.tzinfo is not None
    assert document.extracted_at.utcoffset() == timedelta(hours=5, minutes=30)
    assert document.warnings == []
    assert document.limitations == []
    assert page.number == 1
    assert page.width == 612
    assert page.height == 792
    assert page.rotation == 90
    assert page.warnings == []
    assert box.x0 == -4
    assert box.y0 == -2
    assert box.x1 == 700
    assert box.y1 == 22
    assert box.width == 704
    assert box.height == 24
    assert span.text == "Hello"
    assert span.font_name == "Helvetica"
    assert span.font_size == 12
    assert span.font_flags == 0
    assert span.block_index == 0
    assert span.line_index == 0
    assert span.span_index == 0


def test_text_preserves_unicode_and_whitespace():
    text = " \t café\u00a0\n"
    assert _span(text=text).text == text
    assert _span(text="   ").text == "   "


def test_bounding_box_width_and_height():
    box = BoundingBox(x0=-2, y0=1.5, x1=4, y1=1.5)
    assert box.width == 6
    assert box.height == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x0": 2, "y0": 0, "x1": 1, "y1": 1},
        {"x0": 0, "y0": 2, "x1": 1, "y1": 1},
        {"x0": -1, "y0": -1, "x1": -2, "y1": 0},
        {"x0": float("nan"), "y0": 0, "x1": 1, "y1": 1},
        {"x0": 0, "y0": float("inf"), "x1": 1, "y1": 1},
        {"x0": 0, "y0": 0, "x1": float("-inf"), "y1": 1},
    ],
)
def test_bounding_box_rejects_invalid_coordinates(kwargs):
    with pytest.raises(ValidationError):
        BoundingBox(**kwargs)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_page_accepts_quarter_turn_rotations(rotation):
    assert _page(rotation=rotation).rotation == rotation


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 0},
        {"width": -1},
        {"height": 0},
        {"height": float("inf")},
        {"width": float("nan")},
        {"rotation": 45},
        {"rotation": -90},
        {"rotation": 360},
        {"number": 0},
    ],
)
def test_page_rejects_invalid_dimensions_and_rotation(overrides):
    with pytest.raises(ValidationError):
        _page(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"font_size": -0.1},
        {"font_size": float("nan")},
        {"font_size": float("inf")},
        {"block_index": -1},
        {"line_index": -1},
        {"span_index": -1},
    ],
)
def test_span_rejects_invalid_font_size_and_indices(overrides):
    with pytest.raises(ValidationError):
        _span(**overrides)


def test_span_accepts_zero_font_size_and_raw_flags():
    span = _span(font_size=0, font_flags=-1)
    assert span.font_size == 0
    assert span.font_flags == -1


def test_rejects_invalid_hash_and_naive_timestamp():
    with pytest.raises(ValidationError):
        _document(source_sha256="a" * 63)
    with pytest.raises(ValidationError):
        _document(source_sha256="g" * 64)
    with pytest.raises(ValidationError):
        _document(source_sha256=SHA256 + " ")
    with pytest.raises(ValidationError):
        _document(extracted_at=datetime(2026, 10, 16, 12, 0))

    payload = _document().model_dump(mode="json")
    payload["extracted_at"] = "2026-10-16T12:00:00"
    with pytest.raises(ValidationError):
        Document.model_validate(payload)

    digest = "AB" + "cd" * 31
    assert _document(source_sha256=digest).source_sha256 == digest


def test_rejects_duplicate_page_numbers_and_span_ids():
    with pytest.raises(ValidationError, match="page numbers must be unique"):
        _document(
            pages=[
                _page(number=1, spans=[]),
                _page(number=1, spans=[]),
            ]
        )

    with pytest.raises(ValidationError, match="span identifiers must be unique"):
        _document(
            pages=[
                _page(number=1, spans=[_span(id="same")]),
                _page(number=2, spans=[_span(id="same")]),
            ]
        )

    with pytest.raises(ValidationError, match="span identifiers must be unique"):
        _document(
            pages=[
                _page(
                    spans=[
                        _span(id="same", span_index=0),
                        _span(id="same", span_index=1),
                    ]
                )
            ]
        )


def test_rejects_unsupported_schema_version():
    payload = _document().model_dump(mode="json")
    payload["schema_version"] = "0.3.0"
    with pytest.raises(ValidationError):
        Document.model_validate(payload)


def test_rejects_unknown_fields():
    payload = _document().model_dump(mode="json")
    payload["reading_order"] = []
    with pytest.raises(ValidationError):
        Document.model_validate(payload)

    payload = _document().model_dump(mode="json")
    payload["pages"][0]["role"] = "body"
    with pytest.raises(ValidationError):
        Document.model_validate(payload)

    payload = _document().model_dump(mode="json")
    payload["pages"][0]["spans"][0]["bold"] = True
    with pytest.raises(ValidationError):
        Document.model_validate(payload)

    payload = _document().model_dump(mode="json")
    payload["pages"][0]["spans"][0]["bbox"]["width"] = 10
    with pytest.raises(ValidationError):
        Document.model_validate(payload)


def test_json_round_trip_equality():
    document = _document(
        warnings=["skipped annotation"],
        limitations=["images were not extracted"],
        pages=[
            _page(
                rotation=90,
                warnings=["text extends outside the page"],
                spans=[_span(text=" \t café\u00a0\n")],
            )
        ],
    )
    restored = Document.model_validate_json(document.model_dump_json())
    assert restored == document
    assert restored.extracted_at.isoformat() == document.extracted_at.isoformat()
