"""Shared rich console and themed output helpers for the ACDP CLI.

All CLI commands import from here to keep styling consistent across the
platform — colours, borders, icons, and progress styles are defined once.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

__all__ = [
    "console",
    "err_console",
    "print_banner",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_section",
    "print_result_table",
    "boot_progress",
    "task_progress",
]

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        "platform.accent":   "bold cyan",
        "platform.success":  "bold green",
        "platform.error":    "bold red",
        "platform.warning":  "bold yellow",
        "platform.info":     "dim white",
        "platform.heading":  "bold white",
        "platform.muted":    "dim cyan",
        "platform.agent":    "bold magenta",
        "platform.label":    "bold blue",
    }
)

console     = Console(theme=_THEME)
err_console = Console(stderr=True, theme=_THEME)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""
   ___   ___ ___  ___
  / _ | / __/ _ \/ _ \
 / __ |/ (__/ // / ___/
/_/ |_|\___/____/_/

"""

def print_banner(version: str = "0.1.0") -> None:
    """Print the ACDP ASCII banner and version line."""
    banner_text = Text(_BANNER, style="bold cyan", justify="center")
    subtitle = Text(
        f"  Autonomous Cyber Defense Platform  ·  v{version}\n",
        style="dim cyan",
        justify="center",
    )
    console.print(banner_text)
    console.print(subtitle)
    console.rule(style="cyan dim")
    console.print()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def print_success(message: str) -> None:
    console.print(f"  [platform.success]✔[/]  {message}")


def print_error(message: str) -> None:
    err_console.print(f"  [platform.error]✘[/]  {message}")


def print_warning(message: str) -> None:
    console.print(f"  [platform.warning]⚠[/]  {message}")


def print_info(message: str) -> None:
    console.print(f"  [platform.info]·[/]  {message}")


def print_section(title: str) -> None:
    console.print()
    console.rule(f"[platform.heading]{title}[/]", style="cyan dim")
    console.print()


# ---------------------------------------------------------------------------
# Result table
# ---------------------------------------------------------------------------

def print_result_table(
    title: str,
    rows: list[tuple[str, str]],
    *,
    key_header: str = "Field",
    value_header: str = "Value",
) -> None:
    """Print a two-column key/value result table."""
    table = Table(
        title=title,
        title_style="platform.heading",
        border_style="cyan dim",
        header_style="platform.label",
        show_lines=False,
        expand=False,
        padding=(0, 1),
    )
    table.add_column(key_header,   style="platform.muted",   no_wrap=True)
    table.add_column(value_header, style="platform.accent",  no_wrap=False)

    for key, value in rows:
        table.add_row(key, value)

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Boot progress (spinner per step)
# ---------------------------------------------------------------------------

@contextmanager
def boot_progress() -> Generator[Progress, None, None]:
    """A spinner-based progress display for sequential boot steps.

    Usage::

        with boot_progress() as progress:
            task = progress.add_task("Loading config…", total=None)
            ...
            progress.update(task, description="[green]Config loaded")
    """
    progress = Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    )
    with progress:
        yield progress


# ---------------------------------------------------------------------------
# Ingestion / general task progress (bar)
# ---------------------------------------------------------------------------

@contextmanager
def task_progress(description: str, total: int) -> Generator[Progress, None, None]:
    """A progress-bar display for work with a known total unit count.

    Usage::

        with task_progress("Embedding chunks", total=len(chunks)) as (progress, task):
            for chunk in chunks:
                ...
                progress.advance(task)
    """
    progress = Progress(
        SpinnerColumn(spinner_name="dots2", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id
