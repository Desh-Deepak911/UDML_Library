# UDML

A Python 3.12+ library for a layout-independent PDF document intermediate
representation.

## MVP scope

- Extract PDF text spans, coordinates, and page metadata.
- Serialize versioned, loss-aware JSON with source and parser provenance.
- Provide a simple CLI.

The document model is implemented in `udml.core`. PDF extraction and
serialization still need implementation. Reliable reading order, tables,
sections, and semantic structure reconstruction are not supported. No LLM
or AMC-specific adapters.

## Document model

`Document`, `Page`, `TextSpan`, and `BoundingBox` in `udml.core` are the
intermediate representation. They store extracted physical text, page
geometry, and extraction provenance. This is not a reliable semantic
structure or a complete preservation of the PDF. Spans stay in parser source
order, which is not a reliable reading order. The model has no semantic
labels or inferred relationships.

Coordinates are PDF points (72 points = 1 inch). The origin is the top-left
of the unrotated page. X increases to the right and y increases downward.
`Page.width`, `Page.height`, and each bounding box use that unrotated space.
`Page.rotation` is stored separately as 0, 90, 180, or 270 degrees. Boxes
keep extracted coordinates when content falls outside the page, including
negative values.

`schema_version` is `0.1.0`. Unknown fields are rejected. `source_sha256` is
64 hexadecimal characters. `extracted_at` includes a timezone. Page numbers
are one-based and unique in the document. Span `id` values are unique across
every page. `text` keeps the extracted characters, including spaces.
Document `warnings` and `limitations`, and page `warnings`, record
extraction loss.

## Development

Run these commands from the project directory:

    uv sync
    uv run udml
    uv run pytest
    uv run ruff check .
    uv run ruff format --check .

Keep local PDFs in samples/ and generated results in outputs/.
