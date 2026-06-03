"""Backward-compatible wrapper for the pig behavior CLI.

Prefer installing the package and running ``pig-behavior`` or
``python -m pig_behavior``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Run the package CLI from a source checkout."""
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from pig_behavior.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
