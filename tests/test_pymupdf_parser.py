import hashlib
from datetime import UTC, datetime, timedelta

import pymupdf

from udml.parsers import PyMuPDFParser

_LIMITATIONS = [
    "No OCR is performed.",
    "Images, annotations, and semantic or table structure are not represented.",
    "Extracted text reflects PyMuPDF's decoding, not complete PDF fidelity.",
]


def _assert_recorded_limitations(limitations: list[str]) -> None:
    assert limitations[:3] == _LIMITATIONS
    recorded = "\n".join(limitations)
    assert "unrotated CropBox" in recorded
    assert "TEXT_MEDIABOX_CLIP" in recorded
    assert "may be omitted" in recorded
    assert "TEXT_PRESERVE_LIGATURES" in recorded
    assert "TEXT_PRESERVE_WHITESPACE" in recorded
    assert "TEXT_USE_CID_FOR_UNKNOWN_UNICODE" in recorded


def _save(tmp_path, name, pdf):
    data = pdf.tobytes()
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_extracts_known_text_and_provenance(tmp_path):
    pdf = pymupdf.open()
    try:
        page = pdf.new_page(width=200, height=100)
        page.insert_text((20, 40), "Hello UDML", fontsize=11)
        path = _save(tmp_path, "known.pdf", pdf)
    finally:
        pdf.close()

    before = datetime.now(UTC)
    document = PyMuPDFParser().parse(path)
    after = datetime.now(UTC)

    assert document.source_filename == "known.pdf"
    assert document.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert document.parser_name == "pymupdf"
    assert document.parser_version == pymupdf.VersionBind
    assert document.extracted_at.tzinfo is not None
    assert document.extracted_at.utcoffset() == timedelta(0)
    assert before <= document.extracted_at <= after
    assert document.warnings == []
    _assert_recorded_limitations(document.limitations)
    assert len(document.pages) == 1

    parsed = document.pages[0]
    assert parsed.number == 1
    assert parsed.width == 200
    assert parsed.height == 100
    assert parsed.rotation == 0
    assert parsed.warnings == []
    assert len(parsed.spans) == 1

    span = parsed.spans[0]
    assert span.id == "p1-b0-l0-s0"
    assert span.text == "Hello UDML"
    assert span.font_size == 11
    assert isinstance(span.font_flags, int)
    assert span.font_name != ""
    assert span.block_index == 0
    assert span.line_index == 0
    assert span.span_index == 0


def test_rotated_page_keeps_unrotated_geometry(tmp_path):
    pdf = pymupdf.open()
    try:
        page = pdf.new_page(width=200, height=100)
        page.insert_text((20, 40), "Rotated", fontsize=11)
        raw = page.get_text("dict", sort=False)
        expected = raw["blocks"][0]["lines"][0]["spans"][0]
        page.set_rotation(90)
        data = pdf.tobytes()
    finally:
        pdf.close()

    saved = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert saved[0].rotation == 90
        assert saved[0].rect.width == 100
        assert saved[0].rect.height == 200
    finally:
        saved.close()

    path = tmp_path / "rotated.pdf"
    path.write_bytes(data)
    parsed = PyMuPDFParser().parse(path).pages[0]

    assert parsed.rotation == 90
    assert parsed.width == 200
    assert parsed.height == 100
    assert len(parsed.spans) == 1
    span = parsed.spans[0]
    assert span.text == "Rotated"
    assert (span.bbox.x0, span.bbox.y0, span.bbox.x1, span.bbox.y1) == tuple(
        expected["bbox"]
    )


def test_page_without_text_is_kept_with_warning(tmp_path):
    pdf = pymupdf.open()
    try:
        pdf.new_page(width=300, height=400)
        path = _save(tmp_path, "blank.pdf", pdf)
    finally:
        pdf.close()

    document = PyMuPDFParser().parse(path)

    assert len(document.pages) == 1
    _assert_recorded_limitations(document.limitations)
    parsed = document.pages[0]
    assert parsed.number == 1
    assert parsed.width == 300
    assert parsed.height == 400
    assert parsed.rotation == 0
    assert parsed.spans == []
    assert parsed.warnings == ["No text spans were extracted from this page."]
    warning = parsed.warnings[0].lower()
    assert "scan" not in warning
    assert "ocr" not in warning
