"""Save and load a Document as JSON."""

import os
import tempfile
from pathlib import Path

from udml.core.models import Document


def save_document(
    document: Document,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write ``document`` as indented UTF-8 JSON.

    Missing parent directories are created. An existing destination is left
    unchanged unless ``overwrite`` is true. The destination appears only after
    the full payload has been written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(f"Output file already exists: {destination}")

    payload = document.model_dump_json(indent=2, ensure_ascii=False)
    if not payload.endswith("\n"):
        payload += "\n"

    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=".udml-",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(
                    f"Output file already exists: {destination}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def load_document(path: str | Path) -> Document:
    """Read UTF-8 JSON and validate it as a Document."""
    return Document.model_validate_json(Path(path).read_text(encoding="utf-8"))
