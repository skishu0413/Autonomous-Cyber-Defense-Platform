"""Enables ``python -m acdp`` execution.

Delegates to :func:`acdp.main.main` which supports:

* ``python -m acdp [--config <path>]``   — boot the platform
* ``python -m acdp ingest ...``          — run the ingestion CLI
"""
import sys
from acdp.main import main

sys.exit(main())
