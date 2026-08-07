"""Pulling readable text out of the common binary document formats.

`read_file` used to decode every file as UTF-8. On a PDF that handed back ~a
third replacement characters, which the model narrated as an "internal error"
rather than admitting it could not read it - the owner had asked it to memorise a
résumé and it made up an excuse instead. This module is the fix: a small dispatch
by extension, PDF through `pypdf` and the OOXML trio (`.docx` / `.xlsx` / `.pptx`)
through the standard library alone - they are ZIP archives of XML, and the text is
right there once you unzip it, so they cost no dependency. Formatting is not
preserved; the caller wants the words, not the layout.

`extract_document` returns the text, an empty string for a document that genuinely
has none (a scanned PDF), or `None` when the path is not a format handled here - in
which case `read_file` falls back to its text path, which now refuses an opaque
binary instead of laundering it into gibberish.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from daemon.tools.base import ToolError

MAX_CHARS = 1_000_000
"""Ceiling on extracted text, matching `read_file`'s own memory concern: a
thousand-page PDF should not be pulled into a single string only for `_truncate`
to throw all but a few thousand characters away."""

# OOXML namespaces. Word keeps its text in `<w:t>`, PowerPoint and the shared
# DrawingML in `<a:t>`, spreadsheets in `<t>` under the spreadsheet namespace.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # a stripped install; say how to fix it
        raise ToolError(
            "reading PDF files needs the 'pypdf' package, which is missing here; "
            "install it with `pip install pypdf`"
        ) from exc
    reader = PdfReader(str(path))
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        piece = page.extract_text() or ""
        parts.append(piece)
        total += len(piece)
        if total > MAX_CHARS:
            break
    return "\n".join(parts).strip()


def _extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[str] = []
    for para in root.iter(f"{_W}p"):
        runs = [node.text for node in para.iter(f"{_W}t") if node.text]
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)


def _extract_pptx(path: Path) -> str:
    slides: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        # Numeric order: slide2 before slide10, which lexical sorting gets wrong.
        names.sort(key=lambda n: int(re.search(r"(\d+)", n).group(1)))
        for name in names:
            root = ET.fromstring(archive.read(name))
            runs = [node.text for node in root.iter(f"{_A}t") if node.text]
            if runs:
                slides.append("\n".join(runs))
    return "\n\n".join(slides)


def _extract_xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter(f"{_S}si"):
                shared.append("".join(t.text or "" for t in item.iter(f"{_S}t")))
        rows: list[str] = []
        sheets = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        for sheet in sheets:
            root = ET.fromstring(archive.read(sheet))
            for row in root.iter(f"{_S}row"):
                cells: list[str] = []
                for cell in row.iter(f"{_S}c"):
                    value = cell.find(f"{_S}v")
                    kind = cell.get("t")
                    if kind == "s" and value is not None and value.text is not None:
                        index = int(value.text)
                        cells.append(shared[index] if 0 <= index < len(shared) else "")
                    elif kind == "inlineStr":
                        cells.append("".join(t.text or "" for t in cell.iter(f"{_S}t")))
                    elif value is not None and value.text is not None:
                        cells.append(value.text)
                if cells:
                    rows.append("\t".join(cells))
    return "\n".join(rows)


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".pptx": _extract_pptx,
    ".xlsx": _extract_xlsx,
}


def extract_document(path: Path) -> str | None:
    """Text from a known document format, `""` if it has none, or `None` if the
    suffix is not one handled here (the caller should try the text path)."""
    extractor = _EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        return None
    try:
        return extractor(path)
    except ToolError:
        raise
    except Exception as exc:
        # Broad on purpose: these files come from the owner's disk in whatever
        # shape - truncated, password-locked, mislabelled - and the contract is a
        # clean ToolError, never a crashed turn. pypdf alone raises a dozen types.
        raise ToolError(
            f"{path.name} could not be read as a {path.suffix} document: {exc}"
        ) from exc
