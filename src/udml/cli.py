"""Command-line interface for UDML."""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from udml.core.models import Document
from udml.inspection import InspectionError, inspect_pdf, paths_refer_to_same_file
from udml.parsers import PDFExtractionError, PyMuPDFParser
from udml.serialization import save_document
from udml.structure.tables import RuledGridConfig

# Provisional PDF-point tolerances. They are not calibrated to any document.
_PROVISIONAL_AXIS_TOLERANCE = 0.5
_PROVISIONAL_SNAP_TOLERANCE = 0.5
_PROVISIONAL_GAP_TOLERANCE = 1.0
_PROVISIONAL_CONTAINMENT_TOLERANCE = 1.0


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
    inspect_command.add_argument(
        "--vectors",
        action="store_true",
        help="Also draw stored vector segments",
    )
    inspect_command.add_argument(
        "--grids",
        action="store_true",
        help=(
            "Draw supported ruled-grid cell boxes and highlight unassigned "
            "spans. Results are geometric grid candidates, not confirmed "
            "tables."
        ),
    )
    inspect_command.add_argument(
        "--baseline-fallback",
        action="store_true",
        help=(
            "With --grids, assign a horizontal span when its baseline y is "
            "inside exactly one cell and its box fits that cell horizontally. "
            "This does not show that glyph ink lies inside the cell."
        ),
    )
    _add_grid_tolerance(
        inspect_command,
        "--axis-tolerance",
        _PROVISIONAL_AXIS_TOLERANCE,
        "axis alignment",
    )
    _add_grid_tolerance(
        inspect_command,
        "--snap-tolerance",
        _PROVISIONAL_SNAP_TOLERANCE,
        "coordinate snapping",
    )
    _add_grid_tolerance(
        inspect_command,
        "--gap-tolerance",
        _PROVISIONAL_GAP_TOLERANCE,
        "collinear gap",
    )
    _add_grid_tolerance(
        inspect_command,
        "--containment-tolerance",
        _PROVISIONAL_CONTAINMENT_TOLERANCE,
        "span containment",
    )
    args = parser.parse_args(argv)
    if args.command == "parse":
        _parse(args.pdf, args.output, overwrite=args.overwrite)
        return
    _inspect(
        args.pdf,
        args.model,
        args.output,
        vectors=args.vectors,
        grids=args.grids,
        axis_tolerance=args.axis_tolerance,
        snap_tolerance=args.snap_tolerance,
        gap_tolerance=args.gap_tolerance,
        containment_tolerance=args.containment_tolerance,
        baseline_fallback=args.baseline_fallback,
    )


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


def _add_grid_tolerance(
    command: argparse.ArgumentParser,
    name: str,
    default: float,
    kind: str,
) -> None:
    command.add_argument(
        name,
        type=float,
        default=None,
        metavar="PT",
        help=(
            f"Provisional {kind} tolerance in PDF points "
            f"(default: {default}). Not calibrated. Used only with --grids."
        ),
    )


def _inspect(
    pdf: str,
    model: str,
    output: str,
    *,
    vectors: bool,
    grids: bool,
    axis_tolerance: float | None,
    snap_tolerance: float | None,
    gap_tolerance: float | None,
    containment_tolerance: float | None,
    baseline_fallback: bool,
) -> None:
    try:
        configuration = _grid_configuration(
            grids=grids,
            axis_tolerance=axis_tolerance,
            snap_tolerance=snap_tolerance,
            gap_tolerance=gap_tolerance,
            containment_tolerance=containment_tolerance,
            baseline_fallback=baseline_fallback,
        )
        counts = inspect_pdf(
            pdf,
            model,
            output,
            vectors=vectors,
            grids=configuration,
            baseline_fallback=baseline_fallback,
        )
    except (
        OSError,
        PDFExtractionError,
        ValidationError,
        InspectionError,
        CLIError,
    ) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Diagnostic copy: {output}")
    print(f"Pages: {counts.pages}")
    print(f"Spans: {counts.spans}")
    if vectors:
        print(f"Vector segments: {counts.segments}")
        print(
            "Overlays show stored geometry. They do not prove that "
            "extraction is complete or that visible table boundaries "
            "were found."
        )
    else:
        print(
            "Rectangles show stored span geometry. This is not a faithful "
            "archival reproduction and does not prove that all source content "
            "was extracted."
        )
    if grids:
        print(f"Grid candidates: {counts.candidates}")
        print(f"Supported cells: {counts.cells}")
        if baseline_fallback:
            print(f"Strict assignments: {counts.assigned_spans}")
            print(f"Baseline fallback assignments: {counts.fallback_spans}")
        else:
            print(f"Assigned spans: {counts.assigned_spans}")
        print(f"Unassigned spans: {counts.unassigned_spans}")
        print(
            "Tolerances are provisional and are not calibrated. "
            "Cell boxes are geometric grid candidates, not confirmed tables."
        )
        if baseline_fallback:
            print(
                "Baseline fallback assignments are not evidence that "
                "glyph ink lies inside the cell."
            )


def _grid_configuration(
    *,
    grids: bool,
    axis_tolerance: float | None,
    snap_tolerance: float | None,
    gap_tolerance: float | None,
    containment_tolerance: float | None,
    baseline_fallback: bool,
) -> RuledGridConfig | None:
    supplied = (
        axis_tolerance,
        snap_tolerance,
        gap_tolerance,
        containment_tolerance,
    )
    if not grids:
        if any(value is not None for value in supplied):
            raise CLIError("Grid tolerances require --grids.")
        if baseline_fallback:
            raise CLIError("Baseline fallback requires --grids.")
        return None
    return RuledGridConfig(
        axis_tolerance=_chosen(axis_tolerance, _PROVISIONAL_AXIS_TOLERANCE),
        snap_tolerance=_chosen(snap_tolerance, _PROVISIONAL_SNAP_TOLERANCE),
        gap_tolerance=_chosen(gap_tolerance, _PROVISIONAL_GAP_TOLERANCE),
        containment_tolerance=_chosen(
            containment_tolerance, _PROVISIONAL_CONTAINMENT_TOLERANCE
        ),
    )


def _chosen(value: float | None, default: float) -> float:
    return default if value is None else value


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
