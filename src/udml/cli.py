"""Command-line interface for UDML."""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from udml.core.models import Document
from udml.inspection import InspectionError, inspect_pdf, paths_refer_to_same_file
from udml.parsers import PDFExtractionError, PyMuPDFParser
from udml.serialization import save_document


class CLIError(Exception):
    """An expected command-line failure."""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="udml")
    commands = parser.add_subparsers(dest="command", required=True)
    parse_command = commands.add_parser("parse", help="Extract a PDF to UDML JSON")
    parse_command.add_argument("pdf", help="Source PDF path")
    parse_command.add_argument("--output", required=True, help="JSON output path")
    parse_command.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    inspect_command = commands.add_parser(
        "inspect",
        help="Draw stored span rectangles on a diagnostic PDF copy",
    )
    inspect_command.add_argument("pdf", help="Source PDF path")
    inspect_command.add_argument("--model", required=True, help="UDML JSON path")
    inspect_command.add_argument(
        "--output",
        required=True,
        help="Diagnostic PDF path",
    )
    args = parser.parse_args(argv)
    if args.command == "parse":
        _parse(args.pdf, args.output, overwrite=args.overwrite)
        return
    _inspect(args.pdf, args.model, args.output)


def _parse(pdf: str, output: str, *, overwrite: bool) -> None:
    source = Path(pdf)
    destination = Path(output)
    try:
        # Check again after parsing so --overwrite cannot replace a hard link,
        # symlink, or case alias created while the PDF was being read.
        _reject_if_output_is_source(source, destination)
        document = PyMuPDFParser().parse(source)
        _reject_if_output_is_source(source, destination)
        save_document(document, destination, overwrite=overwrite)
    except (OSError, PDFExtractionError, ValidationError, CLIError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    _print_summary(destination, document)


def _reject_if_output_is_source(source: Path, destination: Path) -> None:
    if paths_refer_to_same_file(source, destination):
        raise CLIError("Output path refers to the source PDF.")


def _inspect(pdf: str, model: str, output: str) -> None:
    try:
        page_count, span_count = inspect_pdf(pdf, model, output)
    except (OSError, PDFExtractionError, ValidationError, InspectionError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Diagnostic copy: {output}")
    print(f"Pages: {page_count}")
    print(f"Spans: {span_count}")
    print(
        "Rectangles show stored span geometry. This is not a faithful "
        "archival reproduction and does not prove that all source content "
        "was extracted."
    )


def _print_summary(path: Path, document: Document) -> None:
    span_count = sum(len(page.spans) for page in document.pages)
    print(f"Output: {path}")
    print(f"Pages: {len(document.pages)}")
    print(f"Spans: {span_count}")
    for warning in document.warnings:
        print(f"Warning: {warning}")
    for page in document.pages:
        for warning in page.warnings:
            print(f"Warning: page {page.number}: {warning}")
