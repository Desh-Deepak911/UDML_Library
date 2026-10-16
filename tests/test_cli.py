import pymupdf

from udml.cli import main
from udml.serialization import load_document


def test_cli_parse_writes_udml_json(tmp_path, capsys):
    pdf = pymupdf.open()
    try:
        page = pdf.new_page(width=200, height=100)
        page.insert_text((20, 40), "Hello UDML", fontsize=11)
        source = tmp_path / "example.pdf"
        source.write_bytes(pdf.tobytes())
    finally:
        pdf.close()

    output = tmp_path / "outputs" / "example.json"
    main(["parse", str(source), "--output", str(output)])

    document = load_document(output)
    captured = capsys.readouterr()
    text = "".join(span.text for page in document.pages for span in page.spans)
    span_count = sum(len(page.spans) for page in document.pages)
    assert document.source_filename == "example.pdf"
    assert document.schema_version == "0.5.0"
    assert len(document.pages) == 1
    assert "Hello UDML" in text
    assert f"Output: {output}" in captured.out
    assert "Pages: 1" in captured.out
    assert f"Spans: {span_count}" in captured.out
