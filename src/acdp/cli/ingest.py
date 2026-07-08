"""Ingestion CLI — load knowledge sources into the vector store.

Usage
-----
Standalone:

    python -m acdp.cli.ingest \\
        --config config.example.yaml \\
        --source-id owasp-genai-v1 \\
        --category owasp_genai \\
        --file /path/to/owasp-genai.txt

Via the acdp package entry point:

    python -m acdp ingest \\
        --config config.example.yaml \\
        --source-id playbook-001 \\
        --category playbook \\
        --file /path/to/playbook.md

Arguments
---------
--config       Path to the YAML configuration file (default: config.yaml or
               config.example.yaml).
--source-id    Unique identifier for the knowledge source (required).
--category     Source category; one of: owasp_genai, mitre_attack,
               mitre_atlas, playbook, compliance, topology, other (required).
--file         Path to the knowledge source file to ingest (required).

Exit codes
----------
0   Ingestion succeeded.
1   Configuration error, ingestion error, or any other failure. The error
    message identifying the cause is printed to stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acdp.exceptions import ConfigError, IngestionError
from acdp.models import KnowledgeSource, SourceCategory

__all__ = ["main"]


def _read_xlsx(file_path: Path) -> str:
    """Extract plain text from an Excel workbook using openpyxl.

    Each sheet is rendered as a tab-separated block of rows, separated by a
    blank line between sheets.

    Args:
        file_path: Path to the .xlsx file.

    Returns:
        Text content extracted from all sheets and cells.

    Raises:
        ImportError: If openpyxl is not installed.
        OSError: If the file cannot be read.
    """
    import openpyxl  # type: ignore[import]

    wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
    sections: list[str] = []
    for sheet in wb.worksheets:
        rows: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(cell) if cell is not None else "" for cell in row]
            # Skip entirely empty rows
            if any(c.strip() for c in cells):
                rows.append("\t".join(cells))
        if rows:
            sections.append(f"=== Sheet: {sheet.title} ===\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(sections)


def _read_pdf(file_path: Path) -> str:
    """Extract plain text from a PDF file using pypdf.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Concatenated text content of all pages.

    Raises:
        ImportError: If pypdf is not installed.
        OSError: If the file cannot be read.
    """
    from pypdf import PdfReader  # type: ignore[import]

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _default_config_path() -> Path:
    """Return config.yaml if it exists, otherwise config.example.yaml."""
    candidate = Path("config.yaml")
    if candidate.exists():
        return candidate
    fallback = Path("config.example.yaml")
    if fallback.exists():
        return fallback
    return candidate


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ingestion CLI.

    Args:
        argv: Command-line arguments (excluding the program name). Defaults to
              ``sys.argv[1:]`` when ``None``.

    Returns:
        Exit code: 0 on success, 1 on any error.
    """
    import argparse
    from acdp.cli.console import (
        console, err_console,
        print_banner, print_success, print_error, print_warning, print_info,
        print_section, print_result_table, boot_progress, task_progress,
    )

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="acdp ingest",
        description="Ingest a knowledge source file into the ACDP vector store.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=(
            "Path to the YAML configuration file "
            "(default: config.yaml or config.example.yaml)"
        ),
    )
    parser.add_argument(
        "--source-id",
        required=True,
        metavar="ID",
        help="Unique identifier for the knowledge source.",
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=[c.value for c in SourceCategory],
        metavar="CATEGORY",
        help=(
            "Source category. One of: "
            + ", ".join(c.value for c in SourceCategory)
        ),
    )
    parser.add_argument(
        "--file",
        required=True,
        metavar="PATH",
        help="Path to the knowledge source file to ingest.",
    )

    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else _default_config_path()
    source_id: str = args.source_id
    category_str: str = args.category
    file_path = Path(args.file)

    print_banner()
    print_section("Knowledge Source Ingestion")

    # --- Load configuration ---
    from acdp.config import ConfigLoader

    with boot_progress() as progress:
        t = progress.add_task("[cyan]Loading configuration[/]…", total=None)
        loader = ConfigLoader()
        try:
            config = loader.load(config_path)
        except ConfigError as exc:
            progress.update(t, description="[red]✘  Configuration failed")
            print_error(f"Configuration error: {exc}")
            return 1
        progress.update(t, description="[green]✔  Configuration loaded", completed=1, total=1)

    # --- Read the knowledge source file ---
    with boot_progress() as progress:
        suffix = file_path.suffix.lower()
        fmt_label = {".pdf": "PDF", ".xlsx": "Excel", ".xlsm": "Excel"}.get(suffix, "text")
        t = progress.add_task(
            f"[cyan]Reading {fmt_label} file[/]  [dim]{file_path.name}[/]…",
            total=None,
        )
        try:
            if suffix == ".pdf":
                content = _read_pdf(file_path)
            elif suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
                content = _read_xlsx(file_path)
            else:
                content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            progress.update(t, description="[red]✘  File not found")
            print_error(f"Source file not found: {str(file_path)!r}")
            return 1
        except OSError as exc:
            progress.update(t, description="[red]✘  File read error")
            print_error(f"Could not read {str(file_path)!r}: {exc}")
            return 1
        except ImportError as exc:
            progress.update(t, description="[red]✘  Missing dependency")
            print_error(
                f"Missing optional dependency — {exc}.\n"
                "  PDF support:   pip install pypdf\n"
                "  Excel support: pip install openpyxl"
            )
            return 1
        progress.update(
            t,
            description=f"[green]✔  File read[/]  [dim]{len(content):,} chars[/]",
            completed=1,
            total=1,
        )

    # --- Connect to vector store ---
    from acdp.llm_gateway import OllamaGateway
    from acdp.knowledge_base.ingest import IngestionPipeline
    from acdp.knowledge_base.store import InMemoryVectorStore, QdrantVectorStore

    llm_gateway = OllamaGateway(config)

    with boot_progress() as progress:
        t = progress.add_task("[cyan]Connecting to vector store[/]…", total=None)
        try:
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(url=config.vector_store_url)
            vector_store = QdrantVectorStore(qdrant_client)
            progress.update(
                t,
                description=f"[green]✔  Qdrant connected[/]  [dim]{config.vector_store_url}[/]",
                completed=1, total=1,
            )
        except Exception as exc:
            progress.update(
                t,
                description="[yellow]⚠  Qdrant unavailable — using in-memory store",
                completed=1, total=1,
            )
            print_warning(f"Qdrant unavailable ({exc}); data will not be persisted.")
            vector_store = InMemoryVectorStore()

    pipeline = IngestionPipeline(gateway=llm_gateway, store=vector_store)

    # --- Ingest ---
    category = SourceCategory(category_str)
    source = KnowledgeSource(
        source_id=source_id,
        category=category,
        content=content,
    )

    with boot_progress() as progress:
        t = progress.add_task(
            f"[cyan]Ingesting[/]  [dim]{source_id}[/]  →  [magenta]{category_str}[/]…",
            total=None,
        )
        try:
            result = pipeline.ingest(source)
        except IngestionError as exc:
            progress.update(t, description="[red]✘  Ingestion failed", completed=1, total=1)
            print_error(f"Ingestion error: {exc}")
            return 1
        except Exception as exc:
            progress.update(t, description="[red]✘  Unexpected error", completed=1, total=1)
            print_error(f"Unexpected error during ingestion: {exc}")
            return 1
        progress.update(
            t,
            description=f"[green]✔  Ingestion complete[/]  [dim]{result.chunk_count} chunks[/]",
            completed=1, total=1,
        )

    # --- Summary ---
    print_result_table(
        "Ingestion Summary",
        [
            ("Source ID",  result.source_id),
            ("Category",   category_str),
            ("File",       str(file_path)),
            ("Format",     fmt_label),
            ("Chunks",     str(result.chunk_count)),
            ("Store",      type(vector_store).__name__),
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
