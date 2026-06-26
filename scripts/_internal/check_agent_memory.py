#!/usr/bin/env python3
"""Check required agent memory files for this repository."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    ROOT / "AGENTS.md",
    ROOT / ".agents" / "memory" / "00_README.md",
    ROOT / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md",
    ROOT / ".agents" / "memory" / "02_CURRENT_DECISION.md",
    ROOT / ".agents" / "memory" / "03_PROJECT_RULES.md",
    ROOT / ".agents" / "memory" / "04_PROJECT_MEMORY_MEDIUM.md",
    ROOT / ".agents" / "memory" / "05_PROJECT_MEMORY_LONG.md",
    ROOT / ".agents" / "memory" / "06_BENCHMARK_NOTES.md",
    ROOT / ".agents" / "memory" / "07_LEGACY_DIFF_NOTES.md",
    ROOT / ".agents" / "memory" / "08_WORKFLOW.md",
]

KEY_PHRASES = [
    "Do not blame weight",
    "Pigs291119_000263_30fps",
    "legacy",
    "hybrid_bytetrack",
    "association.py",
    "all_detection_indices",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    missing_files: list[Path] = []
    found_files: list[Path] = []
    print("Checking agent memory files:")
    for path in REQUIRED_FILES:
        rel = path.relative_to(ROOT)
        if path.exists():
            found_files.append(path)
            print(f"  FOUND   {rel}")
        else:
            missing_files.append(path)
            print(f"  MISSING {rel}")

    combined_text: list[str] = []
    for path in found_files:
        try:
            combined_text.append(path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"Error reading {path.relative_to(ROOT)}: {exc}", file=sys.stderr)
            return 1

    text_blob = "\n".join(combined_text)
    print("\nChecking key phrases:")
    for phrase in KEY_PHRASES:
        status = "FOUND" if phrase in text_blob else "MISSING"
        print(f"  {status:7} {phrase}")

    return 1 if missing_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
