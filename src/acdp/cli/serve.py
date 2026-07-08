"""CLI entry point for ``python -m acdp serve``.

Loads the platform configuration, starts the HTTP Proxy / Guardrail connector,
and blocks until SIGINT or SIGTERM, then performs a graceful shutdown.

Usage::

    python -m acdp serve [--config PATH]

Exit codes:
  0  — clean shutdown (SIGINT/SIGTERM)
  1  — configuration error or startup failure
"""

from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path

__all__ = ["main"]


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
    """Entry point for ``python -m acdp serve``.

    Returns:
        Exit code: 0 on clean shutdown, 1 on error.
    """
    import argparse
    from acdp.cli.console import (
        print_banner, print_error, print_success, print_info,
    )
    from acdp.exceptions import ConfigError
    from acdp.main import Platform

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="acdp serve",
        description="Start the HTTP Proxy / Guardrail connector.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to YAML configuration file (default: config.yaml or config.example.yaml)",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config) if args.config else _default_config_path()

    # Load configuration (errors print to stderr and exit 1 — no banner)
    try:
        platform = Platform.from_config_path(config_path)
    except ConfigError as exc:
        print_error(f"Configuration error: {exc}")
        return 1
    except Exception as exc:
        print_error(f"Startup error: {exc}")
        return 1

    # Only print banner after successful startup
    print_banner()
    print_success(f"GuardrailProxy connector starting  (config: {config_path})")

    stop_event = threading.Event()

    def _handle_signal(signum, frame):
        print_info("Shutdown signal received — stopping connectors…")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        platform.start_connectors()
    except Exception as exc:
        print_error(f"Failed to start connectors: {exc}")
        return 1

    print_success("Connectors started — waiting for SIGINT/SIGTERM")
    stop_event.wait()

    platform.stop_connectors()
    print_success("Connectors stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
