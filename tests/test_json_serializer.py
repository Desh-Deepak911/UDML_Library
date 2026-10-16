from datetime import datetime, timedelta, timezone

from udml.core import BoundingBox, Document, Page, TextSpan
from udml.core.models import build_source_groups
from udml.serialization import load_document, save_document


def test_json_round_trip_preserves_document(tmp_path):
    span = TextSpan(
        id="p2-b1-l2-s3",
        text="  café \n",
        bbox=BoundingBox(x0=-4, y0=-2, x1=20, y1=10),
        font_name="Helvetica",
        font_size=11,
        font_flags=16,
        block_index=1,
        line_index=2,
        span_index=3,
    )
    lines, blocks = build_source_groups(2, [span], {(1, 2): ((1.0, 0.0), 0)})
    document = Document(
        source_filename="sample.pdf",
        source_sha256="ab" * 32,
        parser_name="pymupdf",
        parser_version="1.28.2",
        extracted_at=datetime(
            2026, 10, 16, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
        ),
        pages=[
            Page(
                number=2,
                width=200,
                height=100,
                rotation=90,
                warnings=["No text spans were extracted from this page."],
                spans=[span],
                lines=lines,
                blocks=blocks,
            )
        ],
        warnings=["PyMuPDF repaired this PDF during extraction."],
        limitations=["No OCR is performed."],
    )
    path = tmp_path / "nested" / "doc.json"

    save_document(document, path)
    restored = load_document(path)
    raw = path.read_text(encoding="utf-8")

    assert restored == document
    assert restored.extracted_at.isoformat() == document.extracted_at.isoformat()
    assert "café" in raw
    assert "\\u00e9" not in raw
