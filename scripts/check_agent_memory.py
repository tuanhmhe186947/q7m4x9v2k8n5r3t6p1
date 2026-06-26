from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

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
    "IDSW ≈ 6",
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

    combined_text = []
    for path in found_files:
        try:
            combined_text.append(path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"  ERROR   could not read {path.relative_to(ROOT)}: {exc}")
            missing_files.append(path)

    all_text = "\n".join(combined_text)
    all_text_lower = all_text.lower()
    missing_phrases = [phrase for phrase in KEY_PHRASES if phrase.lower() not in all_text_lower]

    print("\nChecking key phrases:")
    for phrase in KEY_PHRASES:
        status = "FOUND" if phrase not in missing_phrases else "MISSING"
        print(f"  {status:<7} {phrase}")

    if missing_files or missing_phrases:
        if missing_files:
            print("\nWarnings: missing required files detected.")
        if missing_phrases:
            print("\nWarnings: missing required phrases detected.")
        return 1

    print("\nAgent memory check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
