#!/usr/bin/env python3
"""Print the core agent context files used in this repository."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

CONTEXT_FILES = [
    ROOT / "AGENTS.md",
    ROOT / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md",
    ROOT / ".agents" / "memory" / "02_CURRENT_DECISION.md",
    ROOT / ".agents" / "memory" / "03_PROJECT_RULES.md",
    ROOT / ".agents" / "memory" / "08_WORKFLOW.md",
]


def print_file(path: Path) -> None:
    rel = path.relative_to(ROOT)
    print("=" * 80)
    print(rel)
    print("=" * 80)
    if not path.exists():
        print("[missing]\n")
        return
    print(path.read_text(encoding="utf-8").rstrip())
    print("\n")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    for path in CONTEXT_FILES:
        try:
            print_file(path)
        except OSError as exc:
            print(f"Error reading {path.relative_to(ROOT)}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
